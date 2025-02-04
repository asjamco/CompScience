# New dataset: Loan approval prediction
loan_data = [
    ['employed', 'high', 'low', 'approved'],
    ['unemployed', 'low', 'high', 'denied'],
    ['self-employed', 'medium', 'low', 'approved'],
    ['employed', 'low', 'high', 'denied'],
    ['self-employed', 'high', 'low', 'approved'],
    ['unemployed', 'medium', 'high', 'denied'],
    ['employed', 'medium', 'low', 'approved'],
    ['self-employed', 'low', 'high', 'denied'],
    ['unemployed', 'high', 'low', 'approved'],
    ['employed', 'low', 'low', 'approved']
]

# Extract features (X) and target labels (Y)
X_train = [row[:-1] for row in loan_data]
Y_train = [row[-1] for row in loan_data]

# Function to compute prior probabilities P(Y)
def compute_prior(labels):
    total_samples = len(labels)
    prior_probs = {label: labels.count(label) / total_samples for label in set(labels)}
    return prior_probs

# Function to compute likelihood P(X_i | Y)
def compute_likelihood(features, labels):
    feature_probabilities = {}
    num_features = len(features[0])  # Number of feature columns

    for feature_index in range(num_features):
        feature_probabilities[feature_index] = {}

        for class_label in set(labels):
            relevant_samples = [features[i][feature_index] for i in range(len(features)) if labels[i] == class_label]
            total_relevant = len(relevant_samples)
            
            value_counts = {val: relevant_samples.count(val) / total_relevant for val in set(relevant_samples)}
            feature_probabilities[feature_index][class_label] = value_counts

    return feature_probabilities

# Function to compute posterior probability P(Y | X)
def compute_posterior(sample, prior_probs, likelihoods):
    posterior_probs = {}

    for label in prior_probs:
        probability = prior_probs[label]  # Start with the prior probability

        for i, feature_value in enumerate(sample):
            probability *= likelihoods.get(i, {}).get(label, {}).get(feature_value, 0)

        posterior_probs[label] = probability

    return posterior_probs

# Function to predict the class for a new sample
def classify(sample, prior_probs, likelihoods):
    posteriors = compute_posterior(sample, prior_probs, likelihoods)
    return max(posteriors, key=posteriors.get)

# Training: Calculate priors and likelihoods
prior_probs = compute_prior(Y_train)
likelihoods = compute_likelihood(X_train, Y_train)

# Test with a new sample
test_sample = ['self-employed', 'medium', 'high']  # New loan applicant
predicted_label = classify(test_sample, prior_probs, likelihoods)

print(f"The predicted class for {test_sample} is: {predicted_label}")
