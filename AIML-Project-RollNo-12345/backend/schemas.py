from pydantic import BaseModel, Field
from typing import Optional

class StudentProfile(BaseModel):
    gender: str = Field(..., description="M or F")
    ssc_p: float = Field(..., ge=0, le=100, description="SSC Percentage (10th)")
    hsc_p: float = Field(..., ge=0, le=100, description="HSC Percentage (12th)")
    hsc_s: str = Field(..., description="HSC Specialization (Commerce, Science, Arts)")
    degree_p: float = Field(..., ge=0, le=100, description="Degree Percentage")
    degree_t: str = Field(..., description="Degree Type (Sci&Tech, Comm&Mgmt, Others)")
    workex: str = Field(..., description="Work Experience (Yes or No)")
    etest_p: float = Field(..., ge=0, le=100, description="E-Test Percentage")
    specialisation: Optional[str] = Field("None", description="MBA Specialisation (Mkt&HR, Mkt&Fin)")
    mba_p: Optional[float] = Field(0.0, ge=0, le=100, description="MBA Percentage")

class PredictionResponse(BaseModel):
    placement_probability: float
    prediction: str
    recommendations: list[str]
    feature_importance: dict[str, float] = {}
