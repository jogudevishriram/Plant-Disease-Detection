# ----------------- Install dependencies -----------------
# pip install ultralytics python-telegram-bot==13.15 opencv-python-headless pillow

from ultralytics import YOLO
import cv2
from io import BytesIO
import numpy as np
import os
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext, CommandHandler

# ----------------- Load YOLOv8 model -----------------
model = YOLO("best.pt")  # make sure best.pt is the same model as in Colab

# ----------------- Telegram Token -----------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "7829830499:AAGAcAvZTzMcAdzK7LfVEjxGf0DiV9czYzM")

# ----------------- Start command -----------------
def start(update, context):
    update.message.reply_text(
        "🌿 Hi! Send me a plant image, and I will detect the disease with highest confidence and return an annotated image."
    )

# ----------------- Handle incoming images -----------------
def handle_image(update, context: CallbackContext):
    try:
        # Get the image from Telegram
        file = context.bot.getFile(update.message.photo[-1].file_id)
        file_bytes = file.download_as_bytearray()

        # Convert bytes to OpenCV image
        npimg = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Run YOLOv8 prediction (same conf/iou as Colab)
        results = model.predict(source=img, conf=0.1, iou=0.7)

        # Annotate the image (bounding boxes + labels)
        annotated_img = results[0].plot()

        # Find top detection (highest confidence) across all results
        top_detection = None
        top_conf = 0.0
        for r in results:  # loop over all result objects
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                if conf > top_conf:
                    top_detection = (label, conf)
                    top_conf = conf

        # Convert annotated image to bytes
        _, buffer = cv2.imencode('.jpg', annotated_img)
        bio = BytesIO(buffer)
        bio.name = 'result.jpg'
        bio.seek(0)

        # Send result with label + confidence
        if top_detection:
            label, conf = top_detection
            caption = f"✅ Highest confidence disease: {label} (confidence: {conf:.2f})"
            update.message.reply_photo(photo=bio, caption=caption)
        else:
            update.message.reply_photo(photo=bio, caption="❌ No disease detected in the image.")

    except Exception as e:
        update.message.reply_text(f"❌ Error: {e}")

# ----------------- Start the bot -----------------
updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
dispatcher = updater.dispatcher

dispatcher.add_handler(CommandHandler('start', start))
dispatcher.add_handler(MessageHandler(Filters.photo, handle_image))

print("Bot is running...")
updater.start_polling()
updater.idle()
