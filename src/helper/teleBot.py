import os
import telebot


def send_message(text):
    """Simple function to send a message"""
    # Initialize bot with token
    BOT_TOKEN = os.environ.get('BOT_TOKEN')
    CHAT_ID = os.environ.get('CHAT_ID')

    bot = telebot.TeleBot(BOT_TOKEN)
    try:
        bot.send_message(CHAT_ID, f"🔔 {text}")
        return True
    except Exception as e:
        print(f"Error sending message: {e}")
        return False
    
"""
Extra Code for later Bot improvements, that I just won't get to :)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = 
Welcome to the Notification Bot! 👋
Available commands:
/register - Register this chat for notifications
/send <message> - Send a notification message
/status - Check if you're registered

    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['register'])
def register_chat(message):
    global ADMIN_CHAT_ID
    ADMIN_CHAT_ID = message.chat.id
    bot.reply_to(message, f"✅ Successfully registered! Your chat ID is: {ADMIN_CHAT_ID}")

@bot.message_handler(commands=['status'])
def check_status(message):
    if message.chat.id == ADMIN_CHAT_ID:
        bot.reply_to(message, "✅ This chat is registered for notifications!")
    else:
        bot.reply_to(message, "❌ This chat is not registered for notifications.")

@bot.message_handler(commands=['send'])
def send_notification(message):
    if message.chat.id != ADMIN_CHAT_ID:
        bot.reply_to(message, "❌ You're not authorized to send notifications!")
        return
    
    # Extract the message content after the /send command
    try:
        notification_text = message.text.split('/send ', 1)[1]
        if ADMIN_CHAT_ID:
            bot.send_message(ADMIN_CHAT_ID, f"🔔 Notification:\n{notification_text}")
        else:
            bot.reply_to(message, "❌ No chat is registered for notifications yet!")
    except IndexError:
        bot.reply_to(message, "❌ Please include a message after /send")

def send_notification_programmatically(message_text):
    Function to send notifications from your Python code
    if ADMIN_CHAT_ID:
        bot.send_message(ADMIN_CHAT_ID, f"🔔 Notification:\n{message_text}")
        return True
    return False

# Start the bot
def main():
    print("Notification Bot started...")
    send_notification_programmatically("Hello, this is a test notification from your Telegram bot!")
    bot.infinity_polling()

if __name__ == "__main__":
    main()


"""