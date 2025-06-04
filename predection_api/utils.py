from typing import Optional
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext

# --- 1) Gestion simple des utilisateurs en mémoire (exemple) ---
fake_users_db = {
    "alice": {
        "username": "alice",
        # mot de passe “secret” haché avec bcrypt
        "hashed_password": "$2b$12$uNstVN0eVvI1r4mOaczeJO8jn6T.pBGnmpNnXrOIhTqNMTBURvlGK",
        "disabled": False,
    }
}

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_user(db, username: str) -> Optional[dict]:
    return db.get(username)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = get_user(fake_users_db, username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(username: str) -> str:
    # Pour simplifier, on retourne un “fake token” = username
    return username


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    user = get_user(fake_users_db, token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if user.get("disabled"):
        raise HTTPException(status_code=400, detail="Utilisateur inactif")
    return user
