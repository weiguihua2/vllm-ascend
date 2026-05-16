def _model_forward(
        self,
        num_tokens_padded: int,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ):
        assert self.model is not None
        forward_context = get_forward_context()
        assert forward_context is not None

        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
            **model_kwargs,
        }
        if self.tp_rank == 0:
            logger.info(f"======== input_ids: {inputs_embeds}, positions: {positions}")
        run_model = partial(self.model, **model_inputs)

        if self.enable_enpu:
            # The soft segmentation scenario requires event.record first, then event.wait
            self._update_full_graph_params_if_needed(
                forward_context, num_tokens_padded, positions
            )
            hidden_states = run_model()
        else:
            hidden_states = run_model()
            self._update_full_graph_params_if_needed(
                forward_context, num_tokens_padded, positions
            )

        if forward_context.flash_comm_v1_enabled and not isinstance(hidden_states, IntermediateTensors):
            hidden_states = self._all_gather_hidden_states_and_aux(hidden_states)
        return hidden_states
