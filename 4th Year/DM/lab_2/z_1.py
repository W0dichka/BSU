def determine_scale_type(value):
    data_types = {
        "gender": "nominal",
        "age": "ratio",
        "education level": "ordinal",
        "salary": "ratio",
        "temperature": "interval"
    }
    
    return data_types.get(value.lower(), "Unknown type")

data = ["Gender", "Age", "Education level", "Salary", "Temperature"]

for item in data:
    scale_type = determine_scale_type(item)
    print(f"{item}: {scale_type}")
