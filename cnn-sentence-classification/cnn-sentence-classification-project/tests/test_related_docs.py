# # Appears 1,2,3
# topic_prob_vector = [
#     (1, 0.30),
#     (2, 0.15),
#     (3, 0.13),
#     (4, 0.10),
#     (5, 0.10)]

# Appears all
# topic_prob_vector = [
#     (1, 0.10),
#     (2, 0.10),
#     (3, 0.10),
#     (4, 0.10),
#     (5, 0.10)]

# Appears 1,2
topic_prob_vector = [
    (1, 0.30),
    (2, 0.15),
    (3, 0.11),
    (4, 0.10),
    (5, 0.10)]

# topic_prob_vector is ordered by the probability. If the probability of a topic is greater than
# the probability of the next topic * a multiplier, then the next topics are discard.
multipliers_list = [2, 1.3, 1.1, 1]  # List of multipliers applied to topic i+1
topics = []  # List that stores the index of the best topics
for i in range(len(topic_prob_vector)):
    topics.append(topic_prob_vector[i][0])  # Add the current topic index to the list of topics

    # Break if the last topic was reached
    if i == len(topic_prob_vector) - 1:
        break

    # Last multiplier value is used for the topics above len(multipliers_list) position in the topic_prob_vector
    multipliers_index = min(i, len(multipliers_list) - 1)
    multiplier = multipliers_list[multipliers_index]

    # If the prob of the current topic is > than the prob of the next topic * multiplier, discard next topics
    if topic_prob_vector[i][1] > topic_prob_vector[i + 1][1] * multiplier:
        break

print(topics)
