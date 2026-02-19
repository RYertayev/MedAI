from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import mimetypes

from openai import OpenAI

app = Flask(__name__, static_folder="public", static_url_path="/")
CORS(app)

client = OpenAI(
    api_key="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"
)

# Храним последнее распознавание (для демо)
last_context = None  # сюда будем класть последний отчет (анализы или фото)
last_type = None     # "labs" | "face"


@app.route("/")
def index():
    return send_from_directory("public", "index.html")


def _b64_data_url(file_bytes: bytes, filename: str, content_type: str | None):
    """
    Возвращает data URL для OpenAI input_image.
    Поддержим jpg/png/webp/pdf.
    """
    ct = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    b64 = base64.b64encode(file_bytes).decode("utf-8")

    # OpenAI vision принимает input_image с image_url; для PDF чаще лучше конвертировать в картинки,
    # но для прототипа можно пытаться отправлять как data:application/pdf.
    return f"data:{ct};base64,{b64}"


def _safe_str(x):
    return (x or "").strip()


def _common_disclaimer_ru():
    return (
        "⚠️ Важно: это информационная расшифровка и общие рекомендации. "
        "Это НЕ диагноз и не заменяет врача. "
        "При сильных симптомах/ухудшении состояния — обратитесь к врачу или в экстренные службы (103/112)."
    )


@app.route("/analyze/labs", methods=["POST"])
def analyze_labs():
    """
    Принимает файл анализов/справок: PNG/JPG/PDF
    Поле: file (multipart/form-data)
    Доп. поля (необязательно): sex, age
    """
    global last_context, last_type

    if "file" not in request.files:
        return jsonify({"result": "❌ Файл не найден (ожидается поле file)"}), 400

    f = request.files["file"]
    filename = f.filename or "upload"
    content_type = f.content_type
    file_bytes = f.read()

    if not file_bytes:
        return jsonify({"result": "❌ Пустой файл"}), 400

    # лимит для демо (10MB)
    if len(file_bytes) > 10 * 1024 * 1024:
        return jsonify({"result": "❌ Файл слишком большой (максимум 10MB)"}), 413

    sex = _safe_str(request.form.get("sex"))
    age = _safe_str(request.form.get("age"))

    data_url = _b64_data_url(file_bytes, filename, content_type)

    prompt_text = f"""
Ты — MedAI Assistant: система, которая расшифровывает медицинские анализы/справки по изображению/скану.

ЗАДАЧА:
1) Аккуратно извлеки показатели из документа (если это таблица/анализы).
2) Определи отклонения "ниже/выше нормы" (если в документе указаны референсы — используй их как приоритет).
   Если референсы не указаны — используй типовые нормы для взрослых и отметь, что нормы могут отличаться по лаборатории.
3) Дай понятное объяснение простым языком: что может означать отклонение (без постановки диагноза).
4) Сформируй рекомендации "следующие шаги": к какому врачу/что уточнить/какой анализ пересдать.
5) Отдельно выдели "красные флаги" (когда нужно срочно к врачу), но только если действительно видно опасные значения.

ДАННЫЕ О ПОЛЬЗОВАТЕЛЕ (если указаны):
- Пол: {sex if sex else "не указан"}
- Возраст: {age if age else "не указан"}

ОГРАНИЧЕНИЯ И БЕЗОПАСНОСТЬ:
- НЕ ставь диагноз.
- НЕ назначай лекарства и дозировки.
- Пиши осторожно: "возможные причины", "стоит обсудить с врачом".
- Если документ нечитабелен — попроси более четкое фото/скан.

СТРОГИЙ ФОРМАТ ОТВЕТА:

Краткий вывод (1–2 предложения):
...

Таблица показателей:
- <показатель>: <значение> <ед.изм>, норма: <min–max или "не указана">, статус: <норма/ниже/выше>

Что это может означать:
- ...

Рекомендации:
- ...

Красные флаги (если есть):
- ...

{_common_disclaimer_ru()}
""".strip()

    try:
        response = client.responses.create(
            model="gpt-5",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_text},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
        )

        analysis_text = response.output_text
        last_context = analysis_text
        last_type = "labs"

        return jsonify({"result": analysis_text})

    except Exception as e:
        return jsonify({"result": f"❌ Ошибка анализа: {str(e)}"}), 500


@app.route("/analyze/face", methods=["POST"])
def analyze_face():
    """
    Принимает фото лица (селфи) и делает ОБЩУЮ оценку состояния по визуальным признакам.
    Поле: image (multipart/form-data)
    """
    global last_context, last_type

    if "image" not in request.files:
        return jsonify({"result": "❌ Фото не найдено (ожидается поле image)"}), 400

    image_file = request.files["image"]
    filename = image_file.filename or "face.jpg"
    content_type = image_file.content_type or "image/jpeg"
    image_bytes = image_file.read()

    if not image_bytes:
        return jsonify({"result": "❌ Пустое изображение"}), 400

    if len(image_bytes) > 10 * 1024 * 1024:
        return jsonify({"result": "❌ Файл слишком большой (максимум 10MB)"}), 413

    data_url = _b64_data_url(image_bytes, filename, content_type)

    prompt_text = f"""
Ты — MedAI Assistant. Проанализируй селфи/фото лица и дай осторожную, НЕдиагностическую оценку.

ЗАДАЧА:
1) Опиши видимые признаки, которые могут указывать на усталость/недосып/стресс/обезвоживание
   (например: выраженность кругов под глазами, покраснение глаз, бледность, напряженная мимика).
2) Если фото не подходит (не лицо/плохой свет/размыто) — скажи, как переснять.
3) Дай безопасные рекомендации уровня образа жизни: сон, вода, отдых, когда обратиться к врачу.

ОГРАНИЧЕНИЯ:
- НЕ ставь диагноз.
- НЕ утверждай наличие заболеваний.
- НЕ делай выводы о серьезных медицинских состояниях только по фото.
- НЕ назначай лекарства и дозировки.

ФОРМАТ:

Краткая оценка:
...

Что видно на фото:
- ...

Возможные объяснения (осторожно):
- ...

Рекомендации:
- ...

Когда стоит обратиться к врачу:
- ...

{_common_disclaimer_ru()}
""".strip()

    try:
        response = client.responses.create(
            model="gpt-5",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_text},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
        )

        analysis_text = response.output_text
        last_context = analysis_text
        last_type = "face"

        return jsonify({"result": analysis_text})

    except Exception as e:
        return jsonify({"result": f"❌ Ошибка анализа: {str(e)}"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    Чат поверх последнего результата (анализы или лицо).
    JSON: { "question": "..." }
    """
    global last_context, last_type

    if not last_context:
        return jsonify({"answer": "⚠ Сначала загрузите анализы (раздел Загрузка) или сделайте фото (раздел Фото)."}), 400

    data = request.get_json(silent=True) or {}
    question = _safe_str(data.get("question"))

    if not question:
        return jsonify({"answer": "⚠ Напишите вопрос."}), 400

    context_label = "Результаты анализов/справки" if last_type == "labs" else "Результат фото-анализа лица"

    prompt = f"""
Ты — MedAI Assistant. Отвечай на вопросы пользователя по последнему анализу.

{context_label}:
{last_context}

Вопрос пользователя:
{question}

ПРАВИЛА:
- Дай понятный ответ простым языком.
- Если вопрос про диагноз/лечение/дозировки — отвечай безопасно: объясни общие причины и посоветуй врача.
- Не выдумывай показатели, которых нет в контексте.
- В конце добавь короткое предупреждение о том, что это не заменяет врача.

Добавь в конце строку:
{_common_disclaimer_ru()}
""".strip()

    try:
        response = client.responses.create(
            model="gpt-5",
            input=prompt,
        )
        return jsonify({"answer": response.output_text})

    except Exception as e:
        return jsonify({"answer": f"❌ Ошибка: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)