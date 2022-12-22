from pydantic import BaseModel


class senA(BaseModel):
    reviewerName: str
    reviewTitle: str
    reviewBody: str
    reviewStars: int
