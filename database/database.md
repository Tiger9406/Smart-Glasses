# Database Schema rn

uses local **SQLite3** database to persist user identities and their biometric data (facial and voice embeddings). database file is specified by `IDENTITY_DB_PATH` in the core.config (defaulting to `./api/identities.db`).

### 1. `users`
Table for known people:

id: INTEGER PRIMARY KEY AUTOINCREMENT

name: TEXT NOT NULL *note: not enforced as unique*

### 2. `face_embeddings`
Table for facial embeddings

id: INTEGER PRIMARY KEY AUTOINCREMENT

user_id: INTEGER NOT NULL FOREIGN KEY references id in users

embedding: ARRAY NOT NULL (converted from numpy.ndarray)

### 3. `voice_embeddings`
Table for audio embedding

id: INTEGER PRIMARY KEY AUTOINCREMENT

user_id: INTEGER NOT NULL FOREIGN KEY references id in users

embedding: ARRAY NOT NULL (converted form numpy.ndarray)

### 4. `history`
Table for character history

id: INTEGER PRIMARY KEY AUTOINCREMENT

user_id: INTEGER NOT NULL FOREIGN KEY references id in users

transcript: TEXT NOT NULL

timestamp: DATETIME DEFAULT CURRENT_TIMESTAMP

### NOTE

**`ARRAY`**: This is a custom data type mapped in the `DatabaseManager`. When saving data, an `numpy.ndarray` is converted into binary data using `io.BytesIO` and `np.save()`. When retrieving data, the binary blob is read back into a working `numpy.ndarray` using `np.load()`.