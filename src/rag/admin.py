from django.contrib import admin

from rag.models import (
    Answer,
    AnswerSource,
    IndexBuild,
    IndexBuildRevision,
    IndexedChunk,
    IndexProfile,
    Question,
)


class ReadOnlyAdmin(admin.ModelAdmin):
    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(IndexProfile)
class IndexProfileAdmin(ReadOnlyAdmin):
    list_display = ["name", "provider", "model_id", "dimension", "metric", "created_at"]


@admin.register(IndexBuild)
class IndexBuildAdmin(ReadOnlyAdmin):
    list_display = ["id", "profile", "status", "is_active", "chunk_count", "created_at"]
    list_filter = ["status", "is_active", "profile"]


@admin.register(IndexBuildRevision)
class IndexBuildRevisionAdmin(ReadOnlyAdmin):
    list_display = ["build", "document", "revision"]


@admin.register(IndexedChunk)
class IndexedChunkAdmin(ReadOnlyAdmin):
    list_display = ["document", "ordinal", "profile", "revision"]
    list_filter = ["profile"]
    search_fields = ["document__title"]


class AnswerSourceInline(admin.TabularInline):
    model = AnswerSource
    extra = 0
    fields = ["rank", "document_title", "revision_sequence", "ordinal", "score", "cited"]
    readonly_fields = fields

    def has_add_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(Question)
class QuestionAdmin(ReadOnlyAdmin):
    list_display = ["text", "asked_by", "created_at"]
    search_fields = ["text"]


@admin.register(Answer)
class AnswerAdmin(ReadOnlyAdmin):
    list_display = ["question", "status", "model", "latency_ms", "created_at"]
    list_filter = ["status"]
    inlines = [AnswerSourceInline]
