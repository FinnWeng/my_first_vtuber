import chromadb


# the vector DB acts like long term memory
# the short term memory should rely on other 

class Hippocampus:
    def __init__(self, db_path = "../db/") -> None:
        client = chromadb.PersistentClient(path=db_path)

        # collection = client.get_or_create_collection(name="vtb_memory")
        self.collection = client.get_or_create_collection(name="test")
        self.db_size = self.collection.count()
    
    def add_memory(self, msg):
        
        collection.add(documents =[msg], ids=["{}".format(self.db_size)])
        self.db_size = self.collection.count()
    def query_memory(self, msg):
        output = collection.query(query_texts=[msg],n_results=10,include=["documents"])["documents"][0]
        





        



if __name__ == "__main__":
    client = chromadb.PersistentClient(path="../db/")
    collection = client.get_or_create_collection(name="test")
    import pdb
    pdb.set_trace()
    # collection.add(documents =["The SG-027 ZIMMERMAN Shotgun is an Arm unit in ARMORED CORE VI FIRES OF RUBICON."], ids=["3"])
    # collection.query(query_texts=["tell me about little gem"],n_results=10,)
