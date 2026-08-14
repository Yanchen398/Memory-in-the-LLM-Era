def run_memoryos_dispatch(memoryos_entrypoint="main", **kwargs):
    if memoryos_entrypoint == "main_lme":
        from ..dataset_hygiene import resolve_required_endpoint
        from .main_lme import run_memoryos

        embedding_base_url = resolve_required_endpoint(
            kwargs.get("embedding_base_url"),
            env_name="MEMORY_EMBEDDING_BASE_URL",
            purpose="MemoryOS embedding endpoint",
        )

        return run_memoryos(
            dataset_path=kwargs["dataset_path"],
            output_path=kwargs["output_path"],
            memory_path=kwargs["memory_path"],
            llm_model=kwargs["llm_model"],
            llm_api_key=kwargs["llm_api_key"],
            llm_base_url=kwargs["llm_base_url"],
            embedding_model_name=kwargs["embedding_model_name"],
            embedding_api_key=kwargs.get("embedding_api_key", "EMPTY"),
            embedding_base_url=embedding_base_url,
            sample_concurrency=kwargs.get("sample_concurrency", 16),
            sample_max_retries=kwargs.get("sample_max_retries", 2),
        )

    from .supervised_runner import run_memoryos_supervised

    kwargs.pop("memoryos_entrypoint", None)
    return run_memoryos_supervised(**kwargs)
