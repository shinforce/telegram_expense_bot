# --- MAIN APPLICATION LOGIC ---
async def main() -> None:
    """Set up and run the bot."""
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add command and message handlers
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    # Tell Telegram the webhook URL. We use the token as a secret path.
    webhook_path = f"/{TELEGRAM_TOKEN}"
    full_webhook_url = f"{WEBHOOK_URL}{webhook_path}"

    # Perform the async setup
    await application.initialize()
    await application.bot.set_webhook(url=full_webhook_url)
    
    # Start the internal HTTP server to listen for updates
    await application.updater.start_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=webhook_path
    )

    # Start the bot's update processing
    await application.start()

    # Keep the script running
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
