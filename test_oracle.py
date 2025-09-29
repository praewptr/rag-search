#!/usr/bin/env python3
"""
Test script for Oracle database connection and data retrieval.
This script tests the Oracle database service functionality.
"""

import sys

sys.path.append(".")

from services.oracle_db import oracle_service


def test_oracle_connection():
    """Test Oracle database connection and data retrieval."""
    try:
        print("🔍 Testing Oracle database connection...")

        # Test connection
        oracle_service.get_connection()
        print("✅ Successfully connected to Oracle database!")

        # Test data retrieval
        print("\n📊 Fetching knowledge data from Oracle...")
        data = oracle_service.fetch_knowledge_data()

        print(f"✅ Successfully retrieved {len(data['value'])} records!")

        # Show sample data structure
        if data["value"]:
            print("\n📋 Sample record structure:")
            sample_record = data["value"][0]
            for key, value in sample_record.items():
                print(f"  {key}: {type(value).__name__} = {str(value)[:100]}...")

        # Test individual record retrieval
        if data["value"]:
            first_id = data["value"][0].get("id")
            if first_id:
                print(f"\n🔍 Testing individual record retrieval for ID {first_id}...")
                record = oracle_service.get_record_by_id(first_id)
                if record:
                    print("✅ Successfully retrieved individual record!")
                else:
                    print("❌ Failed to retrieve individual record!")

        print("\n🎉 All Oracle database tests passed!")
        return True

    except Exception as e:
        print(f"❌ Oracle database test failed: {str(e)}")
        return False
    finally:
        # Clean up connection
        oracle_service.close_connection()
        print("\n🔌 Database connection closed.")


if __name__ == "__main__":
    success = test_oracle_connection()
    sys.exit(0 if success else 1)
