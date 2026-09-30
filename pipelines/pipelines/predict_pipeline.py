from src.predict import predict_category


text = input("Enter a YouTube title or description: ")

prediction = predict_category(text)

print("Predicted category:", prediction)
