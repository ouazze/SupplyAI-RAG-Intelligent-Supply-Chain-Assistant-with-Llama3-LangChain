"""
SupplyAI Data Loader Module
Handles CSV ingestion and conversion to LangChain Document objects.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document


class SupplyChainDataLoader:
    """
    Loads supply chain CSV data and converts rows into structured text documents.
    Each row becomes a Document with rich metadata for vector retrieval.
    """

    def __init__(self, csv_path: str):
        """
        Initialize the loader with the path to the supply chain CSV.
        
        Args:
            csv_path: Absolute or relative path to supply_data.csv
        """
        self.csv_path = Path(csv_path)
        self.df: Optional[pd.DataFrame] = None
        
    def load_csv(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            
        # ✅ FIX delimiter
        self.df = pd.read_csv(self.csv_path, sep=";")

        # ✅ column mapping
        column_mapping = {
            "stock_level": "current_stock",
            "reorder_point": "reorder_level",
            "supplier_name": "supplier",
            "price_usd": "unit_cost",
            "demand_per_week": "demand_forecast"
        }

        self.df = self.df.rename(columns=column_mapping)

        self._validate_columns()

        self.df = self.df.fillna("Unknown")
        
        return self.df
    
    def _validate_columns(self) -> None:
        """
        Validate that the CSV contains expected supply chain columns.
        Warns if standard columns are missing but does not fail.
        """
        expected_cols = [
            "product_id", "product_name", "category", "current_stock",
            "reorder_level", "lead_time_days", "supplier", "unit_cost",
            "last_restock_date", "demand_forecast", "warehouse_location"
        ]
        
        missing = [col for col in expected_cols if col not in self.df.columns]
        if missing:
            print(f"[WARNING] Missing expected columns: {missing}")
            print(f"[INFO] Available columns: {list(self.df.columns)}")
    
    def _row_to_text(self, row: pd.Series) -> str:
        """
        Convert a single DataFrame row into a rich natural language description.
        This text is what gets embedded and retrieved.
        """
        # Build a comprehensive text representation of the supply record
        text_parts = [
            f"Product: {row.get('product_name', 'Unknown')} (ID: {row.get('product_id', 'N/A')})",
            f"Category: {row.get('category', 'Uncategorized')}",
            f"Current Stock Level: {row.get('current_stock', 'N/A')} units",
            f"Reorder Threshold: {row.get('reorder_level', 'N/A')} units",
            f"Lead Time from Supplier: {row.get('lead_time_days', 'N/A')} days",
            f"Supplier: {row.get('supplier', 'Unknown')}",
            f"Unit Cost: ${row.get('unit_cost', 'N/A')}",
            f"Last Restock Date: {row.get('last_restock_date', 'N/A')}",
            f"Demand Forecast: {row.get('demand_forecast', 'N/A')} units projected",
            f"Warehouse Location: {row.get('warehouse_location', 'Unknown')}",
        ]
        
        # Add risk indicators if present
        stock = pd.to_numeric(row.get('current_stock', 0), errors='coerce') or 0
        reorder = pd.to_numeric(row.get('reorder_level', 0), errors='coerce') or 0
        
        if stock <= reorder:
            text_parts.append(f"RISK ALERT: Stock is at or below reorder level. Restock immediately.")
        elif stock <= reorder * 1.2:
            text_parts.append(f"WARNING: Stock approaching reorder threshold.")
        else:
            text_parts.append(f"STATUS: Stock level healthy.")
            
        return "\n".join(text_parts)
    
    def to_documents(self) -> List[Document]:
        """
        Convert all CSV rows into LangChain Document objects.
        
        Returns:
            List of Document objects ready for embedding
        """
        if self.df is None:
            self.load_csv()
            
        documents = []
        
        for idx, row in self.df.iterrows():
            text = self._row_to_text(row)
            
            # Metadata enables filtering and source tracking
            metadata = {
                "source": str(self.csv_path),
                "row_index": idx,
                "product_id": str(row.get('product_id', 'unknown')),
                "product_name": str(row.get('product_name', 'unknown')),
                "category": str(row.get('category', 'unknown')),
                "current_stock": str(row.get('current_stock', '0')),
                "reorder_level": str(row.get('reorder_level', '0')),
                "supplier": str(row.get('supplier', 'unknown')),
                "warehouse": str(row.get('warehouse_location', 'unknown')),
            }
            
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
            
        print(f"[INFO] Converted {len(documents)} rows into Documents.")
        return documents
    
    def get_inventory_summary(self) -> dict:
        """
        Generate aggregate statistics for the dashboard.
        """
        if self.df is None:
            self.load_csv()
            
        df = self.df
        total_products = len(df)
        
        # Calculate risk metrics
        df['current_stock_num'] = pd.to_numeric(df['current_stock'], errors='coerce').fillna(0)
        df['reorder_level_num'] = pd.to_numeric(df['reorder_level'], errors='coerce').fillna(0)
        
        low_stock = len(df[df['current_stock_num'] <= df['reorder_level_num']])
        at_risk = len(df[df['current_stock_num'] <= df['reorder_level_num'] * 1.5])
        
        return {
            "total_products": total_products,
            "low_stock_items": low_stock,
            "at_risk_items": at_risk,
            "healthy_stock": total_products - at_risk
        }