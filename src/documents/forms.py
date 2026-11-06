from django import forms

from documents.models import Document


class DocumentForm(forms.ModelForm):
    """Admin form for creating a document or replacing its content.

    Content is optional on edit: leaving both fields empty keeps the current
    revision and only updates metadata.
    """

    text = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 20, "dir": "auto"}),
        help_text="Plain text content. Leave empty to keep the current content.",
    )
    file = forms.FileField(
        required=False,
        help_text="DOCX file. Leave empty to keep the current content.",
    )

    class Meta:
        model = Document
        fields = ["title", "description"]

    def clean(self):
        cleaned = super().clean()
        has_text = bool(cleaned.get("text"))
        has_file = bool(cleaned.get("file"))
        if has_text and has_file:
            raise forms.ValidationError("Provide either text or a DOCX file, not both.")
        if self.instance.pk is None and not (has_text or has_file):
            raise forms.ValidationError("Provide text or a DOCX file to create a document.")
        return cleaned
