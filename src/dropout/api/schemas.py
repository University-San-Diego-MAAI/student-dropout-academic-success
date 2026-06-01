"""Pydantic request/response models for the FastAPI application."""

from pydantic import BaseModel, Field


class StudentFeatures(BaseModel):
    """Input features for a single student record."""

    # Demographics
    marital_status: int = Field(..., description="1=single, 2=married, etc.")
    gender: int = Field(..., description="0=female, 1=male")
    age_at_enrollment: int
    nationality: int
    international: int

    # Enrollment
    application_mode: int
    application_order: int
    course: int
    daytime_evening_attendance: int

    # Academic background
    previous_qualification: int
    previous_qualification_grade: float
    admission_grade: float
    displaced: int

    # Socioeconomic
    mothers_qualification: int
    fathers_qualification: int
    mothers_occupation: int
    fathers_occupation: int
    educational_special_needs: int

    # Financial
    debtor: int
    tuition_fees_up_to_date: int
    scholarship_holder: int

    # Semester 1
    curricular_units_1st_sem_credited: int
    curricular_units_1st_sem_enrolled: int
    curricular_units_1st_sem_evaluations: int
    curricular_units_1st_sem_approved: int
    curricular_units_1st_sem_grade: float
    curricular_units_1st_sem_without_evaluations: int

    # Semester 2
    curricular_units_2nd_sem_credited: int
    curricular_units_2nd_sem_enrolled: int
    curricular_units_2nd_sem_evaluations: int
    curricular_units_2nd_sem_approved: int
    curricular_units_2nd_sem_grade: float
    curricular_units_2nd_sem_without_evaluations: int

    # Macroeconomic
    unemployment_rate: float
    inflation_rate: float
    gdp: float


class PredictionResponse(BaseModel):
    """Output prediction response."""

    predicted_outcome: str
    dropout_probability: float
    risk_tier: str
    class_probabilities: dict[str, float]
