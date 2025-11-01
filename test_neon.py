#Test Postgres on Neon

from sqlalchemy import create_engine, text # type: ignore

DATABASE_URL = "postgresql://neondb_owner:npg_RhrKcHOG8I2X@ep-wispy-bird-adacw2th-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

# Create engine 
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    result = conn.execute(text("SELECT version();"))
    print("neon db version  :",result.scalar())