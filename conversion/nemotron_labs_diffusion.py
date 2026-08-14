from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("NemotronLabsDiffusionModel")
class NemotronLabsDiffusionModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DREAM

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_causal_attention(False)
        self.gguf_writer.add_diffusion_shift_logits(False)

        mask_token_id = self.hparams.get("mask_token_id")
        if mask_token_id is not None:
            self.gguf_writer.add_mask_token_id(mask_token_id)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        name = name.replace("encoder.layers.", "model.layers.")
        name = name.replace("encoder.embed_tokens.", "model.embed_tokens.")
        name = name.replace("encoder.norm.", "model.norm.")
        name = name.replace("diffusion_head.", "lm_head.")
        yield from super().modify_tensors(data_torch, name, bid)
