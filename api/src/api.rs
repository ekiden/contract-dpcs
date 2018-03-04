rpc_api! {
    metadata {
        name = dp_credit_scoring;
        version = "0.1.0";
        client_attestation_required = false;
    }

    rpc create(CreateRequest) -> CreateResponse;

    rpc train(TrainingRequest) -> TrainingResponse;

    rpc infer(InferenceRequest) -> InferenceResponse;
}
