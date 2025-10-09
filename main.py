from src.predict import predict_sentiment

# --- Function to display the menu ---
def display_menu():
    """Displays the menu options to the user."""
    print("\n" + "="*50)
    print("MAIN MENU:")
    print("  1. 🎬 Analyze Movie Review")
    print("  2. 👋 Exit")
    print("="*50)

# --- Main Program ---
if __name__ == "__main__":
    print("Initializing Sentiment Analysis CLI application...")
    
    # Load the model and vectorizer once when the application starts

    # Only run the application if the model was loaded successfully

        # Main loop for the menu
    while True:
        display_menu()
        choice = input("Please select an option (1-2): ")

        if choice == '1':
            # Receive review input from the user
            user_input = input("💬 Enter a movie review: ")
            
            # Call the prediction function
            sentiment = predict_sentiment(user_input)
            1
            print(f"   ✨ Prediction Result: {sentiment.upper()}")

        elif choice == '2':
            print("👋 Thank you for using this application!")
            break
        
        else:
            print("⚠️ Invalid option. Please enter 1 or 2.")
    
