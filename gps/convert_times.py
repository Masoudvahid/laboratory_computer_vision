import datetime

a = [1690887862650,
     1690887867950,
     ]
for i in a:
    timestamp = i / 1000
    adjusted_timestamp = timestamp - (3 * 60 * 60)  # Subtracting 3 hours
    dt_object = datetime.datetime.fromtimestamp(adjusted_timestamp)
    formatted_result = dt_object.strftime("%H%M%S.%f")[:-4]
    dt_object = datetime.datetime.fromtimestamp(timestamp)

    print(formatted_result)
