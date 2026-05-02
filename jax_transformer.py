import os
import time

import flax
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
import tiktoken
from flax.nnx.training.metrics import Average

tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab

num_transformer_blocks = 12
seqlen = 1024
embed_dim = 768
num_query_heads = 12
num_kv_heads = 4
feed_forward_dim = 4 * embed_dim
batch_size = 24
dropout_rate = 0.1

max_steps = 600000 * 12 // batch_size
init_learning_rate = 5e-4
weight_decay = 1e-1
top_k = 10
dtype = jnp.bfloat16
param_dtype = jnp.float32


def scaled_dot_product_attention(
    Q: jax.Array, K: jax.Array, V: jax.Array, mask: jax.Array | None = None
) -> jax.Array:
    dK = K.shape[-1]

    scores = Q @ K.swapaxes(-2, -1) / dK
    if mask is not None:
        scores = jnp.where(mask == 0, -9e15, scores)
    attn_output = nnx.softmax(scores, axis=-1) @ V

    return attn_output


class GroupedQueryAttention(nnx.Module):
    def __init__(
        self, d_model: int, num_query_heads: int, num_kv_heads: int, rngs: nnx.Rngs
    ) -> None:
        """
        Args:
            d_model: Dimension of the model (e.g., 512, 768, etc.)
            num_query_heads: Number of query heads (e.g., 32)
            num_kv_heads: Number of key-value heads (e.g., 8)
                         Must divide num_query_heads evenly
        """

        assert d_model % num_query_heads == 0, (
            "d_model must be divisible by num_query_heads"
        )
        assert num_query_heads % num_kv_heads == 0, (
            "num_query_heads must be divisible by num_kv_heads"
        )

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads

        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.kv_dim = num_kv_heads * self.head_dim

        # Query projection: still projects to full d_model
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)

        # Key and Value projections: project to fewer dimensions
        # Only num_kv_heads worth of dimensions instead of num_query_heads
        self.k_proj = nnx.Linear(d_model, self.kv_dim, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, self.kv_dim, rngs=rngs)

        # Output projection
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)

        # Scaling factor
        self.scale = jnp.sqrt(self.head_dim)

        # Attention mask

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        batch_size, seq_length, _ = x.shape

        # Apply linear projections
        Q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        K = self.k_proj(x)  # (batch_size, seq_len, kv_dim) <- Smaller!
        V = self.v_proj(x)  # (batch_size, seq_len, kv_dim) <- Smaller!

        # Reshape queries for multiple heads
        Q = Q.reshape(
            batch_size, seq_length, self.num_query_heads, self.head_dim
        ).swapaxes(1, 2)
        # Shape: (batch_size, num_query_heads, seq_len, head_dim)

        # Reshape keys and values for fewer heads
        K = K.reshape(
            batch_size, seq_length, self.num_kv_heads, self.head_dim
        ).swapaxes(1, 2)
        V = V.reshape(
            batch_size, seq_length, self.num_kv_heads, self.head_dim
        ).swapaxes(1, 2)
        # Shape: (batch_size, num_kv_heads, seq_len, head_dim)

        # Repeat K and V to match the number of query heads
        # Each KV head is shared by num_queries_per_kv query heads
        K = K.repeat(self.num_queries_per_kv, axis=1)
        V = V.repeat(self.num_queries_per_kv, axis=1)
        # Shape: (batch_size, num_query_heads, seq_len, head_dim)

        scores = Q @ K.swapaxes(-2, -1) / self.scale
        scores = jnp.where(mask == 0, -9e15, scores)

        attn_output = nnx.softmax(scores, axis=-1) @ V
        attn_output = attn_output.swapaxes(1, 2).reshape(
            batch_size, seq_length, self.d_model
        )

        output = self.out_proj(attn_output)
        return output


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        ff_dim: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ) -> None:
        self.layer_norm1 = nnx.LayerNorm(
            epsilon=1e-6,
            num_features=embed_dim,
            rngs=rngs,
        )
        self.attention = GroupedQueryAttention(
            d_model=embed_dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            rngs=rngs,
        )
        self.dropout1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

        self.layer_norm2 = nnx.LayerNorm(
            epsilon=1e-6,
            num_features=embed_dim,
            rngs=rngs,
        )
        self.linear1 = nnx.Linear(
            in_features=embed_dim,
            out_features=ff_dim,
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            in_features=ff_dim,
            out_features=embed_dim,
            rngs=rngs,
        )
        self.dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array, is_training: bool = True) -> jax.Array:
        seq_len = x.shape[1]
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))

        x1 = self.layer_norm1(x)
        x1 = self.attention(x1, mask=mask)
        x1 = x + self.dropout1(x1, deterministic=not is_training)

        x2 = self.layer_norm2(x1)
        x2 = self.linear1(x2)
        x2 = nnx.gelu(x2)
        x2 = self.linear2(x2)
        x2 = x1 + self.dropout2(x2, deterministic=not is_training)

        return x2


class TokenAndPositionEmbedding(nnx.Module):
    def __init__(
        self, seq_length: int, vocab_size: int, embed_dim: int, rngs: nnx.Rngs
    ) -> None:
        self.token_embed = nnx.Embed(
            num_embeddings=vocab_size, features=embed_dim, rngs=rngs
        )
        self.pos_embed = nnx.Embed(
            num_embeddings=vocab_size, features=embed_dim, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> tuple[nnx.Embed, jax.Array]:
        positions = jnp.arange(0, x.shape[1])[None:]
        position_embedding = self.pos_embed(positions)

        token_embedding = self.token_embed(x)
        return (self.token_embed, token_embedding + position_embedding)


class GPT2(nnx.Module):
    def __init__(
        self,
        seqlen: int,
        vocab_size: int,
        embed_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        rate: float,
        feed_forward_dim: int,
        num_transformer_blocks: int,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ) -> None:
        self.embedding_layer = TokenAndPositionEmbedding(
            seqlen, vocab_size, embed_dim, rngs=rngs
        )
        self.dropout = nnx.Dropout(rate=rate, rngs=rngs)

        self.transformer_blocks = nnx.List(
            [
                TransformerBlock(
                    embed_dim,
                    num_query_heads,
                    num_kv_heads,
                    feed_forward_dim,
                    dropout_rate,
                    rngs=rngs,
                )
                for _ in range(num_transformer_blocks)
            ]
        )

        self.layer_norm = nnx.LayerNorm(epsilon=1e-6, num_features=embed_dim, rngs=rngs)

        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.seqlen = seqlen

    def __call__(self, x: jax.Array, is_training: bool = True) -> jax.Array:
        token_embed, x = self.embedding_layer(x)
        x = self.dropout(x, deterministic=not is_training)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, is_training=is_training)

        x = self.layer_norm(x)
        output = token_embed.attend(x)

        return output

    @nnx.jit
    def sample_from(self, logits: jax.Array, top_k: int = 10) -> jax.Array:
        logits, indices = jax.lax.top_k(logits, k=top_k)
        logits = nnx.softmax(logits)

        output = jax.random.choice(random.PRNGKey(0), indices, p=logits)
        return output

    @nnx.jit
    def generate_step(self, padded_tokens: jax.Array, sample_index: int) -> jax.Array:
        logits = self(padded_tokens)
        next_token = self.sample_from(logits[0][sample_index])
        return next_token

    def generate_text(self, max_token: int, start_tokens: list[int]) -> str:
        generated = []
        print(self.tokenizer.decode(start_tokens), flush=True, end="")
        for i in range(max_token):
            sample_index = len(start_tokens) + len(generated) - 1
            # TODO: use attention masking for better efficiency
            padded_tokens = jnp.array(
                (
                    start_tokens
                    + generated
                    + [0] * (seqlen - len(start_tokens) - len(generated))
                )
            )[None, :]
            next_token = int(self.generate_step(padded_tokens, sample_index))
            if (
                next_token
                == self.tokenizer.encode(
                    "<|endoftext|>", allowed_special={"<|endoftext|>"}
                )[0]
            ):
                break
            generated.append(next_token)
            # decode and print next_token
            print(self.tokenizer.decode([next_token]), flush=True, end="")
        return self.tokenizer.decode(start_tokens + generated)


@nnx.jit
def loss_fn(
    model: GPT2, batch: tuple[jax.Array, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    logits = model(batch[0])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch[1]
    ).mean()

    return loss, logits


@nnx.jit
def train_step(
    model: GPT2,
    optimizer: nnx.ModelAndOptimizer,
    metrics: Average,
    batch: tuple[jax.Array, jax.Array],
) -> None:

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, label=batch[1])
    optimizer.update(grads)


data_dir = "data/"
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


# From: https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/train.py#L116
def get_batch(train_or_eval="train") -> tuple[jax.Array, jax.Array]:

    data = train_data if train_or_eval == "train" else val_data

    ix = np.random.randint(0, len(data) - seqlen, (batch_size,))
    x = np.stack([(data[i : i + seqlen]).astype(np.int64) for i in ix])
    y = np.stack([(data[i + 1 : i + 1 + seqlen]).astype(np.int64) for i in ix])

    return jnp.asarray(x), jnp.asarray(y)


rngs = nnx.Rngs(0)
model = GPT2(
    seqlen,
    vocab_size,
    embed_dim,
    num_query_heads,
    num_kv_heads,
    dropout_rate,
    feed_forward_dim,
    num_transformer_blocks,
    dropout_rate,
    rngs=rngs,
)

schedule = optax.cosine_decay_schedule(
    init_value=init_learning_rate, decay_steps=max_steps
)
optax_chain = optax.chain(
    optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
)


optimizer = nnx.ModelAndOptimizer(model, optax_chain)

train_metrics = nnx.metrics.Average("loss")
val_metrics = nnx.metrics.Average("val_loss")

rng = jax.random.PRNGKey(0)

# start_prompt = "Once upon a time"
# start_tokens = tokenizer.encode(start_prompt)[:seqlen]
# print(f"Initial generated text:")
# generated_text = model.generate_text(seqlen // 10, start_tokens)


metrics_history = {"train_loss": [], "val_loss": []}

step = 0
start_time = time.time()
while True:
    input_batch, target_batch = get_batch("train")
    print(input_batch.shape)
    train_step(model, optimizer, train_metrics, (input_batch, target_batch))

    if step % 200 == 0:
        train_loss = float(train_metrics.compute())
        metrics_history["train_loss"].append(train_loss)

        elapsed_time = time.time() - start_time
        print(
            f"Step {step + 1}, Training loss: {train_loss}, Elapsed Time: {elapsed_time:.2f} seconds"
        )

        # eval step
        input_val_batch, target_val_batch = get_batch("val")
        loss, logits = loss_fn(
            model,
            (input_val_batch, target_val_batch),
        )
        val_metrics.update(val_loss=loss, logits=logits)
        val_loss = float(val_metrics.compute())
        metrics_history["val_loss"].append(val_loss)
        print(f"Step {step + 1}, Validation loss: {val_loss}")
        train_metrics.reset()
        val_metrics.reset()

        start_time = time.time()
    step += 1

    if step > max_steps:
        break

# Final text generation
print(f"Final generated text:")
generated_text = model.generate_text(seqlen // 10, start_tokens)
