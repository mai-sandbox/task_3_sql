import sqlite3
import requests

def test_chinook_db():
    """Test that the Chinook database loads correctly."""
    print("Fetching Chinook database...")
    sql_url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(sql_url)
    response.raise_for_status()
    
    print("Creating in-memory database...")
    conn = sqlite3.connect(":memory:")
    conn.executescript(response.text)
    
    cursor = conn.cursor()
    
    # Test query to count customers
    cursor.execute("SELECT COUNT(*) FROM Customer")
    customer_count = cursor.fetchone()[0]
    print(f"Number of customers: {customer_count}")
    
    # Test query to get top albums
    cursor.execute("""
        SELECT Album.Title, COUNT(Track.TrackId) as TrackCount
        FROM Album
        JOIN Track ON Album.AlbumId = Track.AlbumId
        GROUP BY Album.AlbumId
        ORDER BY TrackCount DESC
        LIMIT 5
    """)
    
    print("\nTop 5 albums by track count:")
    for row in cursor.fetchall():
        print(f"  - {row[0]}: {row[1]} tracks")
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"\nTables in database: {[t[0] for t in tables]}")
    
    conn.close()
    print("\nDatabase test completed successfully!")

if __name__ == "__main__":
    test_chinook_db()