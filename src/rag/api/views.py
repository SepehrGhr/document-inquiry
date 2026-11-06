from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.exceptions import NotFound
from rest_framework.generics import ListAPIView, RetrieveAPIView
from rest_framework.response import Response

from rag import services
from rag.api.serializers import QuestionCreateSerializer, QuestionSerializer
from rag.exceptions import IndexNotReady, InvalidQuestion
from rag.models import Question


class QuestionListCreateView(ListAPIView):
    serializer_class = QuestionSerializer
    queryset = Question.objects.prefetch_related("answers__sources")

    @extend_schema(request=QuestionCreateSerializer, responses={201: QuestionSerializer})
    def post(self, request):
        serializer = QuestionCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            answer = services.ask(text=serializer.validated_data["question"], actor=request.user)
        except InvalidQuestion as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except IndexNotReady as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        question = Question.objects.prefetch_related("answers__sources").get(pk=answer.question_id)
        return Response(QuestionSerializer(question).data, status=status.HTTP_201_CREATED)


class QuestionDetailView(RetrieveAPIView):
    serializer_class = QuestionSerializer
    queryset = Question.objects.prefetch_related("answers__sources")
    lookup_url_kwarg = "question_id"

    def get_object(self):
        question = self.get_queryset().filter(pk=self.kwargs["question_id"]).first()
        if question is None:
            raise NotFound("Question not found.")
        return question
