"""FastAPI routes for health check and prediction."""

from fastapi import APIRouter, HTTPException, Request

from dropout.api.schemas import PredictionResponse, StudentFeatures
from dropout.features.engineer import add_derived_features_dict
from dropout.models.predictor import Predictor

router = APIRouter()


def get_predictor(request: Request) -> Predictor:
    """Retrieve Predictor instance from application state."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are not loaded. Run the training script first.",
        )
    return predictor


@router.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: StudentFeatures, request: Request) -> PredictionResponse:
    """Predict dropout risk for a single student record."""
    predictor: Predictor = get_predictor(request)
    features = payload.model_dump()
    features = _rename_to_dataset_cols(features)
    features = add_derived_features_dict(features)
    try:
        result = predictor.predict(features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PredictionResponse(**result)


def _rename_to_dataset_cols(d: dict) -> dict:
    """Map snake_case API fields back to original dataset column names."""
    mapping = {
        "marital_status": "Marital Status",
        "gender": "Gender",
        "age_at_enrollment": "Age at enrollment",
        "nationality": "Nacionality",
        "international": "International",
        "application_mode": "Application mode",
        "application_order": "Application order",
        "course": "Course",
        "daytime_evening_attendance": "Daytime/evening attendance",
        "previous_qualification": "Previous qualification",
        "previous_qualification_grade": "Previous qualification (grade)",
        "admission_grade": "Admission grade",
        "displaced": "Displaced",
        "mothers_qualification": "Mother's qualification",
        "fathers_qualification": "Father's qualification",
        "mothers_occupation": "Mother's occupation",
        "fathers_occupation": "Father's occupation",
        "educational_special_needs": "Educational special needs",
        "debtor": "Debtor",
        "tuition_fees_up_to_date": "Tuition fees up to date",
        "scholarship_holder": "Scholarship holder",
        "curricular_units_1st_sem_credited": "Curricular units 1st sem (credited)",
        "curricular_units_1st_sem_enrolled": "Curricular units 1st sem (enrolled)",
        "curricular_units_1st_sem_evaluations": "Curricular units 1st sem (evaluations)",
        "curricular_units_1st_sem_approved": "Curricular units 1st sem (approved)",
        "curricular_units_1st_sem_grade": "Curricular units 1st sem (grade)",
        "curricular_units_1st_sem_without_evaluations": (
            "Curricular units 1st sem (without evaluations)"
        ),
        "curricular_units_2nd_sem_credited": "Curricular units 2nd sem (credited)",
        "curricular_units_2nd_sem_enrolled": "Curricular units 2nd sem (enrolled)",
        "curricular_units_2nd_sem_evaluations": "Curricular units 2nd sem (evaluations)",
        "curricular_units_2nd_sem_approved": "Curricular units 2nd sem (approved)",
        "curricular_units_2nd_sem_grade": "Curricular units 2nd sem (grade)",
        "curricular_units_2nd_sem_without_evaluations": (
            "Curricular units 2nd sem (without evaluations)"
        ),
        "unemployment_rate": "Unemployment rate",
        "inflation_rate": "Inflation rate",
        "gdp": "GDP",
    }
    return {mapping[k]: v for k, v in d.items() if k in mapping}
