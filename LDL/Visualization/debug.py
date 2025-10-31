import lmdb
import pickle

# Path to your LMDB file
lmdb_file = 'data/rwave-1024/rwave-1024.lmdb'

# Open the LMDB environment
env = lmdb.open(lmdb_file, readonly=True, lock=False, readahead=False, meminit=False)

# Create a cursor to iterate over the database
with env.begin(write=False) as txn:
    cursor = txn.cursor()

    # Iterate over the database entries
    for key, value in cursor:
        # Print the key and value
        print(f"Key: {key.decode()}")

        # Unpickle the value
        try:
            data = pickle.loads(value)
            print("Data unpickled successfully!")
            # You can print or inspect the data here
        except Exception as e:
            print(f"Error unpickling data: {e}")

# Close the LMDB environment
env.close()
