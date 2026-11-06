from rest_framework import serializers

from documents.models import Document, DocumentRevision


class RevisionSummarySerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentRevision
        fields = [
            "id",
            "sequence",
            "source",
            "original_filename",
            "file_size",
            "content_sha256",
            "parser_version",
            "created_at",
        ]


class RevisionSerializer(RevisionSummarySerializer):
    download_url = serializers.SerializerMethodField()

    class Meta(RevisionSummarySerializer.Meta):
        fields = [*RevisionSummarySerializer.Meta.fields, "text", "download_url"]

    def get_download_url(self, revision) -> str | None:
        if not revision.file:
            return None
        request = self.context.get("request")
        path = f"/api/v1/documents/{revision.document_id}/revisions/{revision.sequence}/file/"
        return request.build_absolute_uri(path) if request else path


class DocumentListSerializer(serializers.ModelSerializer):
    current_revision = serializers.IntegerField(read_only=True)
    current_source = serializers.CharField(read_only=True)

    class Meta:
        model = Document
        fields = [
            "id",
            "title",
            "description",
            "current_revision",
            "current_source",
            "created_at",
            "updated_at",
        ]


class DocumentDetailSerializer(DocumentListSerializer):
    text = serializers.SerializerMethodField()

    class Meta(DocumentListSerializer.Meta):
        fields = [*DocumentListSerializer.Meta.fields, "text"]

    def get_text(self, document) -> str | None:
        revision = document.revisions.order_by("-sequence").first()
        return revision.text if revision else None


class ContentFieldsMixin(serializers.Serializer):
    text = serializers.CharField(required=False, allow_blank=False, trim_whitespace=False)
    file = serializers.FileField(required=False)

    def validate(self, attrs):
        has_text = "text" in attrs
        has_file = "file" in attrs
        if has_text == has_file:
            raise serializers.ValidationError("Provide exactly one of 'text' or 'file'.")
        return attrs


class DocumentCreateSerializer(ContentFieldsMixin):
    title = serializers.CharField(max_length=255)
    description = serializers.CharField(required=False, allow_blank=True, default="")


class DocumentUpdateSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=255, required=False)
    description = serializers.CharField(required=False, allow_blank=True)

    def validate(self, attrs):
        if not attrs:
            raise serializers.ValidationError("Provide at least one field to update.")
        return attrs


class ContentReplaceSerializer(ContentFieldsMixin):
    expected_revision = serializers.IntegerField(min_value=1)
