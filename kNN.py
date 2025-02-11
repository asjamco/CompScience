def sqrt(n, precision=1e-10):
    """Manually calculates the square root using Newton’s method."""
    if n == 0:
        return 0
    guess = n / 2.0
    while True:
        new_guess = (guess + n / guess) / 2
        if abs(new_guess - guess) < precision:
            return new_guess
        guess = new_guess

def Euclidean_Distance(x1, y1, x2, y2):
    """Calculates Euclidean distance without using math.sqrt()."""
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def KNN(datas, guess):
    """Finds the closest point and assigns its color to the guess."""
    lowest = [0, 0, 0, float('inf')]

    for data in datas:
        data[3] = Euclidean_Distance(guess[0], guess[1], data[0], data[1])
    
    for data in datas:
        if data[3] < lowest[3]:
            lowest = data
    
    guess[2] = lowest[2]
    print(f"Guess Color: {guess[2]}")

# **New dataset (changed values)**
data = [
    [35, 15, "Green", 0],
    [45, 55, "Yellow", 0],
    [65, 85, "Yellow", 0],
    [15, 30, "Green", 0],
    [75, 65, "Yellow", 0],
    [55, 20, "Green", 0],
    [20, 75, "Yellow", 0]
]
guess = [25, 40, ""]

KNN(data, guess)
