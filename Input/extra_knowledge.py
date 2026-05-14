from experiment_pipeline.inputs import NotImplementedInput


class ExtraKnowledgeInput(NotImplementedInput):
    def __init__(self) -> None:
        super().__init__("extra_knowledge")


__all__ = ["ExtraKnowledgeInput"]

