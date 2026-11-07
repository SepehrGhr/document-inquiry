from django.http import FileResponse
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.exceptions import NotFound
from rest_framework.generics import ListCreateAPIView, RetrieveUpdateDestroyAPIView
from rest_framework.response import Response
from rest_framework.views import APIView

from documents import selectors, services
from documents.api.serializers import (
    ContentReplaceSerializer,
    DocumentCreateSerializer,
    DocumentDetailSerializer,
    DocumentListSerializer,
    DocumentUpdateSerializer,
    RevisionSerializer,
    RevisionSummarySerializer,
)
from documents.exceptions import DocumentNotFound, InvalidDocumentInput, RevisionConflict


def _content_input(validated):
    return services.ContentInput(text=validated.get("text"), docx_file=validated.get("file"))


def _active_document(document_id):
    document = selectors.get_active_document(document_id)
    if document is None:
        raise NotFound("Document not found.")
    return document


class DocumentListCreateView(ListCreateAPIView):
    def get_queryset(self):
        queryset = selectors.with_current_revision()
        search = self.request.query_params.get("search")
        if search:
            queryset = queryset.filter(title__icontains=search)
        return queryset

    def get_serializer_class(self):
        return DocumentCreateSerializer if self.request.method == "POST" else DocumentListSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        try:
            document = services.create_document(
                title=data["title"],
                description=data.get("description", ""),
                content=_content_input(data),
                actor=request.user,
            )
        except InvalidDocumentInput as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        document = selectors.with_current_revision().get(pk=document.pk)
        return Response(
            DocumentDetailSerializer(document, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )


class DocumentDetailView(RetrieveUpdateDestroyAPIView):
    lookup_url_kwarg = "document_id"

    def get_queryset(self):
        return selectors.with_current_revision()

    def get_object(self):
        return _active_document(self.kwargs["document_id"])

    def get_serializer_class(self):
        if self.request.method == "PATCH":
            return DocumentUpdateSerializer
        return DocumentDetailSerializer

    def retrieve(self, request, *args, **kwargs):
        document = self.get_object()
        return Response(DocumentDetailSerializer(document, context={"request": request}).data)

    def patch(self, request, *args, **kwargs):
        serializer = DocumentUpdateSerializer(data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        try:
            services.update_metadata(
                document_id=self.kwargs["document_id"],
                title=serializer.validated_data.get("title"),
                description=serializer.validated_data.get("description"),
                actor=request.user,
            )
        except DocumentNotFound as exc:
            raise NotFound(str(exc)) from exc
        except InvalidDocumentInput as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        document = selectors.with_current_revision().get(pk=self.kwargs["document_id"])
        return Response(DocumentDetailSerializer(document, context={"request": request}).data)

    def delete(self, request, *args, **kwargs):
        try:
            services.delete_document(document_id=self.kwargs["document_id"], actor=request.user)
        except DocumentNotFound as exc:
            raise NotFound(str(exc)) from exc
        return Response(status=status.HTTP_204_NO_CONTENT)


class RevisionListCreateView(APIView):
    @extend_schema(responses=RevisionSummarySerializer(many=True))
    def get(self, request, document_id):
        document = _active_document(document_id)
        revisions = selectors.revisions(document)
        return Response(RevisionSummarySerializer(revisions, many=True).data)

    @extend_schema(request=ContentReplaceSerializer, responses={201: RevisionSerializer})
    def post(self, request, document_id):
        serializer = ContentReplaceSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        try:
            revision = services.replace_content(
                document_id=document_id,
                content=_content_input(data),
                expected_revision=data["expected_revision"],
                actor=request.user,
            )
        except DocumentNotFound as exc:
            raise NotFound(str(exc)) from exc
        except RevisionConflict as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_409_CONFLICT)
        except InvalidDocumentInput as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(
            RevisionSerializer(revision, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )


class RevisionDetailView(APIView):
    @extend_schema(responses=RevisionSerializer)
    def get(self, request, document_id, sequence):
        document = _active_document(document_id)
        revision = document.revisions.filter(sequence=sequence).first()
        if revision is None:
            raise NotFound("Revision not found.")
        return Response(RevisionSerializer(revision, context={"request": request}).data)


class RevisionFileView(APIView):
    @extend_schema(
        responses={
            (200, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"): bytes
        }
    )
    def get(self, request, document_id, sequence):
        document = _active_document(document_id)
        revision = document.revisions.filter(sequence=sequence).first()
        if revision is None or not revision.file:
            raise NotFound("File not found.")
        response = FileResponse(
            revision.file.open("rb"),
            as_attachment=True,
            filename=revision.original_filename or "document.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        response["X-Content-Type-Options"] = "nosniff"
        return response
