from django.contrib import admin, messages
from django.shortcuts import redirect

from documents import selectors, services
from documents.exceptions import DocumentError
from documents.forms import DocumentForm
from documents.models import Document, DocumentRevision


class DocumentRevisionInline(admin.TabularInline):
    model = DocumentRevision
    extra = 0
    can_delete = False
    fields = ["sequence", "source", "original_filename", "file_size", "created_at", "created_by"]
    readonly_fields = fields
    ordering = ["-sequence"]

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    form = DocumentForm
    list_display = ["title", "current_revision_display", "created_at", "updated_at"]
    search_fields = ["title", "description"]
    readonly_fields = ["created_at", "updated_at", "created_by", "deleted_at", "deleted_by"]
    inlines = [DocumentRevisionInline]

    def get_queryset(self, request):
        return selectors.with_current_revision(super().get_queryset(request))

    @admin.display(description="Revision")
    def current_revision_display(self, document):
        return document.current_revision

    def get_fields(self, request, obj=None):
        if obj is None:
            return ["title", "description", "text", "file"]
        return ["title", "description", "text", "file", *self.readonly_fields]

    def get_readonly_fields(self, request, obj=None):
        return self.readonly_fields if obj else []

    def save_model(self, request, obj, form, change):
        content = services.ContentInput(
            text=form.cleaned_data.get("text") or None,
            docx_file=form.cleaned_data.get("file") or None,
        )
        if not change:
            document = services.create_document(
                title=obj.title,
                description=obj.description,
                content=content,
                actor=request.user,
            )
            obj.pk = document.pk
            return
        services.update_metadata(
            document_id=obj.pk,
            title=obj.title,
            description=obj.description,
            actor=request.user,
        )
        if content.text is not None or content.docx_file is not None:
            current = selectors.current_revision(obj)
            services.replace_content(
                document_id=obj.pk,
                content=content,
                expected_revision=current.sequence,
                actor=request.user,
            )

    def get_deleted_objects(self, objs, request):
        # Revisions are retained by logical document deletion, so they are not
        # part of the deletion cascade the Admin permission check inspects.
        return [], {}, set(), []

    def delete_model(self, request, obj):
        services.delete_document(document_id=obj.pk, actor=request.user)

    def delete_queryset(self, request, queryset):
        for document_id in queryset.values_list("pk", flat=True):
            services.delete_document(document_id=document_id, actor=request.user)

    def changeform_view(self, request, object_id=None, form_url="", extra_context=None):
        try:
            return super().changeform_view(request, object_id, form_url, extra_context)
        except DocumentError as exc:
            self.message_user(request, str(exc), level=messages.ERROR)
            return redirect(request.path)


@admin.register(DocumentRevision)
class DocumentRevisionAdmin(admin.ModelAdmin):
    list_display = ["document", "sequence", "source", "created_at"]
    list_filter = ["source"]
    search_fields = ["document__title"]
    readonly_fields = [
        "document",
        "sequence",
        "text",
        "source",
        "file",
        "original_filename",
        "file_size",
        "content_sha256",
        "parser_version",
        "created_by",
        "created_at",
    ]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


admin.site.site_header = "Document Q&A"
admin.site.site_title = "Document Q&A"
admin.site.index_title = "Document management"
admin.site.empty_value_display = "—"
admin.site.site_url = None
