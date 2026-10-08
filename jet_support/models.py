from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Text, Numeric
from sqlalchemy.dialects.postgresql import UUID, JSONB

import uuid

# from sqlalchemy.orm import relationship

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    func,
    Boolean,
    Text,
    ForeignKey,
    Date,
    Numeric,
    UniqueConstraint,
    Float,
    Index,
    JSON,
    CheckConstraint,
    event,
    select,
)
from sqlalchemy.orm import relationship

Base = declarative_base()

class SearchHistory(Base):
    __tablename__ = "search_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    query = Column(Text, nullable=False, index=True)
    ai_response = Column(Text, nullable=False)
    user_id = Column(Text)


class AircraftListing(Base):
    __tablename__ = "aircraft_listings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    listing_id = Column(String(50), nullable=False, index=True)
    airframe_serial_number = Column(String(100), nullable=False)
    manufacturer = Column(String(100), nullable=False, index=True)
    model = Column(String(100), nullable=False, index=True)
    model_year = Column(Integer, nullable=False, index=True)
    airframe_total_time = Column(Numeric(10, 2))
    left_engine_hours_since_new = Column(Numeric(10, 2))
    right_engine_hours_since_new = Column(Numeric(10, 2))
    left_engine_hours_since_overhauled = Column(Numeric(10, 2))
    right_engine_hours_since_overhauled = Column(Numeric(10, 2))
    date_hours_last_updated = Column(DateTime)
    date_listed = Column(DateTime, index=True)
    days_on_market = Column(Integer)
    listing_broker = Column(String(200))
    seller = Column(String(200))
    physical_location = Column(String(100), index=True)
    asking_price = Column(Numeric(15, 2), index=True)
    airframe_total_time_value_adjustment = Column(Numeric(15, 6))
    engine_maintenance_program_value_adjustment = Column(Numeric(15, 11))
    auxiliary_power_unit_value_adjustment = Column(Numeric(15, 2))
    damage_devaluation = Column(Numeric(15, 6))
    interior_value_adjustment = Column(Numeric(15, 2))
    new_interior_value_adjustment = Column(Numeric(15, 2))
    paint_value_adjustment = Column(Numeric(15, 2))
    new_paint_value_adjustment = Column(Numeric(15, 2))
    airframe_total_time_engine_adjustment = Column(Numeric(15, 8))
    total_options_and_adjustments_to_value = Column(Numeric(15, 2))
    base_value_for_model_year = Column(Numeric(15, 2))
    adjusted_value = Column(Numeric(15, 2))
    guardian_jet_estimated_value = Column(Numeric(15, 4))
    options = Column(JSONB, nullable=True)

    configurations = relationship("AircraftConfiguration", back_populates="listing")


class AircraftConfiguration(Base):
    __tablename__ = "aircraft_configurations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    aircraft_listing_id = Column(UUID(as_uuid=True), ForeignKey("aircraft_listings.id"), nullable=False, index=True)
    category = Column(String(100), nullable=False)
    feature = Column(String(200), nullable=False)
    price_adjustment = Column(Numeric(15, 4))

    listing = relationship("AircraftListing", back_populates="configurations")


class SoldAircraftListing(Base):
    __tablename__ = "sold_aircraft_listings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    listing_id = Column(String(50), nullable=False, index=True)
    airframe_serial_number = Column(String(100), nullable=False)
    manufacturer = Column(String(100), nullable=False, index=True)
    model = Column(String(100), nullable=False, index=True)
    model_year = Column(Integer, nullable=False, index=True)
    airframe_total_time = Column(Numeric(10, 2))
    left_engine_hours_since_new = Column(Numeric(10, 2))
    right_engine_hours_since_new = Column(Numeric(10, 2))
    left_engine_hours_since_overhauled = Column(Numeric(10, 2))
    right_engine_hours_since_overhauled = Column(Numeric(10, 2))
    date_hours_last_updated = Column(DateTime)
    date_listed = Column(DateTime, index=True)
    date_sold = Column(DateTime, index=True)
    days_on_market = Column(Integer)
    listing_broker = Column(String(200))
    seller = Column(String(200))
    physical_location = Column(String(100), index=True)
    asking_price = Column(Numeric(15, 2), index=True)
    sold_price = Column(Numeric(15, 2), index=True)
    airframe_total_time_value_adjustment = Column(Numeric(15, 6))
    engine_maintenance_program_value_adjustment = Column(Numeric(15, 11))
    auxiliary_power_unit_value_adjustment = Column(Numeric(15, 2))
    damage_devaluation = Column(Numeric(15, 6))
    interior_value_adjustment = Column(Numeric(15, 2))
    new_interior_value_adjustment = Column(Numeric(15, 2))
    paint_value_adjustment = Column(Numeric(15, 2))
    new_paint_value_adjustment = Column(Numeric(15, 2))
    airframe_total_time_engine_adjustment = Column(Numeric(15, 8))
    total_options_and_adjustments_to_value = Column(Numeric(15, 4))
    base_value_for_model_year = Column(Numeric(15, 4))
    adjusted_value = Column(Numeric(15, 4))
    relative_value = Column(Numeric(15, 4))
    average_relative_value = Column(String(20))
    options = Column(JSONB, nullable=True)

    configurations = relationship("SoldAircraftConfiguration", back_populates="sold_listing")


class SoldAircraftConfiguration(Base):
    __tablename__ = "sold_aircraft_configurations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    aircraft_listing_id = Column(UUID(as_uuid=True), ForeignKey("sold_aircraft_listings.id"), nullable=False, index=True)
    category = Column(String(100), nullable=False)
    feature = Column(String(200), nullable=False)
    price_adjustment = Column(Numeric(15, 4))

    sold_listing = relationship("SoldAircraftListing", back_populates="configurations")
