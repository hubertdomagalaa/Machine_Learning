from enum import Enum

class Sex(Enum):
    MALE = 1
    FEMALE = 0

class SmokingStatus(Enum):
    SMOKER = 1
    NON_SMOKER = 0

class Patient:
    def __init__(self, name, age, sex, height, weight, num_of_children, smoker):
        self.name = name
        self.set_age(age)
        self.set_sex(sex)
        self.set_height(height)
        self.set_weight(weight)
        self.set_num_children(num_of_children)
        self.set_smoking_status(smoker)

    def set_age(self, age):
        if 0 <= age <= 120:
            self.age = age
        else:
            raise ValueError("Age must be between 0 and 120")

    def set_sex(self, sex):
        if isinstance(sex, Sex):
            self.sex = sex
        else:
            raise ValueError("Sex must be a valid Sex enum value")

    def set_height(self, height):
        if height > 0:
            self.height = height
        else:
            raise ValueError("Height must be positive")

    def set_weight(self, weight):
        if weight > 0:
            self.weight = weight
        else:
            raise ValueError("Weight must be positive")

    def set_num_children(self, num_children):
        if num_children >= 0:
            self.num_of_children = num_children
        else:
            raise ValueError("Number of children must be non-negative")

    def set_smoking_status(self, smoking_status):
        if isinstance(smoking_status, SmokingStatus):
            self.smoker = smoking_status
        else:
            raise ValueError("Smoking status must be a valid SmokingStatus enum value")

    def calculate_bmi(self):
        return self.weight / ((self.height / 100) ** 2)

    def estimated_insurance_cost(self):
        bmi = self.calculate_bmi()
        estimated_cost = 250 * self.age - 128 * self.sex.value + 370 * bmi + 425 * self.num_of_children + 24000 * self.smoker.value - 12500
        print(f"{self.name}'s estimated insurance cost is ${estimated_cost:.2f}.")
        return estimated_cost

    def update_age(self, new_age):
        self.set_age(new_age)
        print(f"{self.name} is now {self.age} years old.")
        self.estimated_insurance_cost()

    def update_num_children(self, new_num_children):
        self.set_num_children(new_num_children)
        if new_num_children == 0:
            print(f"{self.name} does not have any children.")
        elif new_num_children == 1:
            print(f"{self.name} has 1 child.")
        else:
            print(f"{self.name} has {self.num_of_children} children.")

    def patient_profile(self):
        return {
            'name': self.name,
            'age': self.age,
            'sex': self.sex.name,
            'height': self.height,
            'weight': self.weight,
            'bmi': self.calculate_bmi(),
            'num_of_children': self.num_of_children,
            'smoker': self.smoker.name
        }

    def update_smoking_status(self, new_smoking_status):
        self.set_smoking_status(new_smoking_status)
        print(f"{self.name} has changed smoking status to: {self.smoker.name}")

# Example usage
try:
    patient1 = Patient("John Doe", 25, Sex.MALE, 175, 70, 0, SmokingStatus.NON_SMOKER)
    print(patient1.patient_profile())
    patient1.estimated_insurance_cost()
    patient1.update_age(26)
    patient1.update_num_children(1)
    patient1.update_smoking_status(SmokingStatus.SMOKER)
except ValueError as e:
    print(f"Error: {e}")