"""Physical legacy embedding layouts for migration fixtures."""

from cpp_context_engine.storage.sqlite import _execute_script


def materialize_v24_embeddings(connection):
    if not connection.execute(
        "SELECT 1 FROM sqlite_schema WHERE name='embedding_vectors' AND type='view'"
    ).fetchone():
        return
    vectors = [tuple(row) for row in connection.execute("SELECT * FROM embedding_vectors")]
    attachments = [tuple(row) for row in connection.execute("SELECT * FROM variant_embeddings")]
    connection.commit()
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        _execute_script(
            connection,
            """
            DROP VIEW variant_embeddings;
            DROP VIEW embedding_vectors;
            DROP TABLE embedding_attachment_records;
            DROP TABLE embedding_content_records;
            DROP TABLE embedding_namespaces;
            CREATE TABLE embedding_vectors (
                project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                model TEXT NOT NULL, configuration_id TEXT NOT NULL,
                dimensions INTEGER NOT NULL CHECK(dimensions > 0),
                content_hash TEXT NOT NULL, content_text TEXT NOT NULL,
                magnitude REAL NOT NULL CHECK(magnitude > 0),
                vector_encoding INTEGER NOT NULL CHECK(vector_encoding IN (0,1)),
                vector BLOB NOT NULL CHECK(typeof(vector)='blob'),
                PRIMARY KEY(project_id,model,configuration_id,dimensions,content_hash),
                CHECK((vector_encoding=0 AND length(vector)=dimensions*8)
                    OR (vector_encoding=1 AND length(vector)>0))
            );
            CREATE TABLE variant_embeddings (
                project_id INTEGER NOT NULL, variant_id TEXT NOT NULL,
                model TEXT NOT NULL, configuration_id TEXT NOT NULL,
                dimensions INTEGER NOT NULL, content_hash TEXT NOT NULL,
                PRIMARY KEY(project_id,variant_id,model,configuration_id),
                FOREIGN KEY(project_id,variant_id)
                    REFERENCES symbol_variants(project_id,id) ON DELETE CASCADE,
                FOREIGN KEY(project_id,model,configuration_id,dimensions,content_hash)
                    REFERENCES embedding_vectors(
                        project_id,model,configuration_id,dimensions,content_hash)
            ) WITHOUT ROWID;
            CREATE INDEX variant_embeddings_content ON variant_embeddings(
                project_id,model,configuration_id,dimensions,content_hash);
            """,
        )
        connection.executemany("INSERT INTO embedding_vectors VALUES(?,?,?,?,?,?,?,?,?)", vectors)
        connection.executemany("INSERT INTO variant_embeddings VALUES(?,?,?,?,?,?)", attachments)
        connection.execute("PRAGMA user_version=24")
