from datetime import datetime
from typing import Any, Dict, List, Optional

import oracledb
from fastapi import HTTPException

from config import (
    oracle_db_host,
    oracle_db_password,
    oracle_db_port,
    oracle_db_service_name,
    oracle_db_user,
)


class OracleDBService:
    def __init__(
        self, user: str, password: str, host: str, port: int, service_name: str
    ):
        """
        Initialize Oracle database connection parameters.

        Args:
            user: Database user
            password: Database password
            host: Database host
            port: Database port
            service_name: Database service name
        """
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.service_name = service_name
        self._connection = None

    def get_connection(self):
        """
        Get or create database connection.

        Returns:
            oracledb.Connection: Active database connection
        """
        if self._connection is None or not self._connection:
            try:
                self._connection = oracledb.connect(
                    user=self.user,
                    password=self.password,
                    host=self.host,
                    port=self.port,
                    service_name=self.service_name,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to connect to Oracle database: {str(e)}",
                )
        return self._connection

    def close_connection(self):
        """Close the database connection if it exists."""
        if self._connection:
            try:
                self._connection.close()
                self._connection = None
            except Exception as e:
                print(f"Error closing Oracle connection: {e}")

    def fetch_knowledge_data(self, added: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch data from test.knowledge table with optional filtering.

        Args:
            added: Optional filter for ADDED column (0 for pending, 1 for uploaded, None for all)

        Returns:
            Dict containing 'value' key with list of documents (only mapped fields)
        """
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # Build query with optional WHERE clause
            if added is not None:
                cursor.execute("SELECT k.*, k.ROWID FROM DIGITAL_TEST.TBL_KNOWLEDGE k WHERE ADDED = :1", [added])
            else:
                cursor.execute("SELECT k.*, k.ROWID FROM DIGITAL_TEST.TBL_KNOWLEDGE k")

            # Get column names
            columns = [col[0] for col in cursor.description]

            # Fetch and convert rows to dicts
            rows = cursor.fetchall()
            data = []


            for i, row in enumerate(rows):
                row_dict = {}
                for col_name, value in zip(columns, row):
                    if isinstance(value, datetime):
                        value = value.isoformat() + "Z"
                    row_dict[col_name.upper()] = value

                mapped_dict = {
                    "content": row_dict.get("CONTENT"),
                    "source": row_dict.get("USER_NAME"),
                    "timestamp": row_dict.get("CREATED_DATE"),
                    "added": row_dict.get("ADDED"),
                    "id": row_dict.get("ID", i + 1)
                }

                data.append(mapped_dict)

            cursor.close()

            return {"value": data}

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch data from Oracle database: {str(e)}",
            )

    def delete_knowledge_record(self, record_id: int) -> bool:
        """
        Delete a record from DIGITAL_TEST.TBL_KNOWLEDGE table.

        Args:
            record_id: ID of the record to delete

        Returns:
            bool: True if deletion was successful
        """
        try:
            connection = self.get_connection()
            cursor = connection.cursor()

            # Delete the record by ID
            cursor.execute(
                "DELETE FROM DIGITAL_TEST.TBL_KNOWLEDGE WHERE ID = :1", [record_id]
            )

            # Commit the transaction
            connection.commit()

            # Check if any row was affected
            rows_affected = cursor.rowcount
            cursor.close()

            return rows_affected > 0

        except Exception as e:
            # Rollback in case of error
            if self._connection:
                self._connection.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete record from Oracle database: {str(e)}",
            )

    def get_record_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific record by ID.

        Args:
            record_id: ID of the record to fetch

        Returns:
            Dict containing the record data or None if not found
        """
        try:
            connection = self.get_connection()
            cursor = connection.cursor()

            # Execute the query for specific ID
            cursor.execute(
                "SELECT k.*, k.ROWID FROM DIGITAL_TEST.TBL_KNOWLEDGE k WHERE ID = :1",
                [record_id],
            )

            # Get column names
            columns = [col[0] for col in cursor.description]

            # Fetch the row
            row = cursor.fetchone()

            if not row:
                cursor.close()
                return None

            row_dict = {}
            for col_name, value in zip(columns, row):
                if isinstance(value, datetime):
                    value = value.isoformat() + "Z"
                row_dict[col_name] = value

            row_dict["id"] = record_id
            cursor.close()

            return row_dict

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch record from Oracle database: {str(e)}",
            )

    def mark_document_uploaded(self, record_id: int) -> bool:
        """
        Mark a document as uploaded to Azure Search (set ADDED = 1).

        Args:
            record_id: The ID of the record to mark as uploaded

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            connection = self.get_connection()
            cursor = connection.cursor()

            # Update ADDED column to 1
            cursor.execute(
                """
                UPDATE DIGITAL_TEST.TBL_KNOWLEDGE
                SET ADDED = 1
                WHERE ID = :id
                """,
                {"id": record_id},
            )

            # Check if any rows were updated
            rows_updated = cursor.rowcount
            connection.commit()
            cursor.close()

            return rows_updated > 0

        except Exception as e:
            print(f"Error marking document {record_id} as uploaded: {e}")
            try:
                connection.rollback()
            except Exception:
                pass
            return False

    def mark_documents_uploaded_batch(self, record_ids: List[int]) -> Dict[str, int]:
        """
        Mark multiple documents as uploaded to Azure Search (set ADDED = 1).

        Args:
            record_ids: List of record IDs to mark as uploaded

        Returns:
            dict: Statistics about the update operation
        """
        try:
            connection = self.get_connection()
            cursor = connection.cursor()

            # Prepare batch update
            update_data = [{"id": record_id} for record_id in record_ids]

            cursor.executemany(
                """
                UPDATE DIGITAL_TEST.TBL_KNOWLEDGE
                SET ADDED = 1
                WHERE ID = :id
                """,
                update_data,
            )

            rows_updated = cursor.rowcount
            connection.commit()
            cursor.close()

            return {
                "updated_count": rows_updated,
                "failed_count": len(record_ids) - rows_updated,
                "total_requested": len(record_ids),
            }

        except Exception as e:
            print(f"Error in batch update: {e}")
            try:
                connection.rollback()
            except Exception:
                pass
            return {
                "updated_count": 0,
                "failed_count": len(record_ids),
                "total_requested": len(record_ids),
            }

    def __del__(self):
        """Cleanup: close connection when object is destroyed."""
        self.close_connection()


# Create a global instance that can be imported
oracle_service = OracleDBService(
    user=oracle_db_user,
    password=oracle_db_password,
    host=oracle_db_host,
    port=oracle_db_port,
    service_name=oracle_db_service_name,
)
