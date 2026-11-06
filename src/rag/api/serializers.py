from rest_framework import serializers

from rag.models import Answer, AnswerSource, Question


class QuestionCreateSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=2000, trim_whitespace=True)


class AnswerSourceSerializer(serializers.ModelSerializer):
    document_id = serializers.UUIDField(read_only=True)
    revision_id = serializers.UUIDField(read_only=True)

    class Meta:
        model = AnswerSource
        fields = [
            "document_id",
            "document_title",
            "revision_id",
            "revision_sequence",
            "ordinal",
            "rank",
            "score",
            "cited",
            "text",
        ]


class AnswerSerializer(serializers.ModelSerializer):
    sources = AnswerSourceSerializer(many=True, read_only=True)
    build_id = serializers.UUIDField(read_only=True)

    class Meta:
        model = Answer
        fields = [
            "id",
            "status",
            "text",
            "reason",
            "build_id",
            "provider",
            "model",
            "prompt_tokens",
            "completion_tokens",
            "latency_ms",
            "created_at",
            "sources",
        ]


class QuestionSerializer(serializers.ModelSerializer):
    answer = serializers.SerializerMethodField()

    class Meta:
        model = Question
        fields = ["id", "text", "created_at", "answer"]

    def get_answer(self, question) -> dict | None:
        answer = question.answers.order_by("-created_at").first()
        return AnswerSerializer(answer).data if answer else None
