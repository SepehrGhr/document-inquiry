import re
import unicodedata

import django.contrib.postgres.indexes
import django.contrib.postgres.search
from django.db import migrations, models


def populate_search_text(apps, schema_editor):
    # Frozen lexical-1 normalization; changing the live normalizer requires a
    # new backfill migration, not an edit to this historical transformation.
    translation = str.maketrans("يك٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹\u200c\u200d", "یک01234567890123456789  ")
    chunks = apps.get_model("rag", "IndexedChunk").objects.using(schema_editor.connection.alias)
    pending = []
    for chunk in chunks.only("id", "text").iterator(chunk_size=500):
        text = unicodedata.normalize("NFKC", chunk.text).lower().translate(translation)
        text = re.sub(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]", "", text.replace("ـ", ""))
        chunk.search_text = " ".join(text.split())
        pending.append(chunk)
        if len(pending) == 500:
            chunks.bulk_update(pending, ["search_text"])
            pending.clear()
    if pending:
        chunks.bulk_update(pending, ["search_text"])


class Migration(migrations.Migration):
    dependencies = [
        ("documents", "0001_initial"),
        ("rag", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="indexedchunk",
            name="search_text",
            field=models.TextField(default="", editable=False),
        ),
        migrations.RunPython(populate_search_text, migrations.RunPython.noop),
        migrations.AddField(
            model_name="indexedchunk",
            name="search_vector",
            field=models.GeneratedField(
                db_persist=True,
                expression=django.contrib.postgres.search.SearchVector(
                    "search_text", config="simple"
                ),
                output_field=django.contrib.postgres.search.SearchVectorField(),
            ),
        ),
        migrations.AddIndex(
            model_name="indexedchunk",
            index=django.contrib.postgres.indexes.GinIndex(
                fields=["search_vector"], name="chunk_search_vector_gin"
            ),
        ),
    ]
