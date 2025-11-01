from sqlalchemy import Column, Integer, String, DateTime, func
from database import Base

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(String)
    bot_response = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
