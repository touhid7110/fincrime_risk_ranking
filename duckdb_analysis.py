import duckdb
con = duckdb.connect("aml.duckdb")
#con.execute("CREATE TABLEtransactions AS SELECT * FROM read_csv_auto('HI-Small_Trans_cleaned.csv', header=true)")
print(con.execute("SELECT * FROM transactions LIMIT 5").df())   # inspect the real header
print(con.execute("SELECT COUNT(*) FROM transactions").df())

#find the percentage of transactions that are fraudulent
#fraudulent_count = con.execute('SELECT COUNT(*) from ttransactions WHERE is_laundering = 1').df().iloc[0, 0]
#total_count = con.execute('SELECT COUNT (*) from ttransactions').df().iloc[0, 0]
#percentage = (fraudulent_count / total_count) * 100
#print(f"Percentage of fraudulent transactions: {percentage:.2f}%")

# --- Build account_features table ---
con.execute('''
    CREATE OR REPLACE TABLE account_features AS

    WITH sender AS (
        SELECT
            from_account                                                            AS account,
            COUNT(*)                                                                AS sender_tx_count,
            SUM(amount_paid)                                                        AS sender_total_amount,
            AVG(amount_paid)                                                        AS sender_avg_amount,
            MAX(amount_paid)                                                        AS sender_max_amount,
            COUNT(DISTINCT to_account)                                              AS sender_distinct_receivers,
            COUNT(DISTINCT payment_format)                                          AS sender_distinct_formats,
            COUNT(DISTINCT payment_currency)                                        AS sender_distinct_currencies,
            ROUND(SUM(CASE WHEN amount_paid % 1 = 0 THEN 1 ELSE 0 END)
                  * 100.0 / COUNT(*), 2)                                            AS sender_pct_round_amounts
        FROM transactions
        GROUP BY from_account
    ),

    receiver AS (
        SELECT
            to_account                                                              AS account,
            COUNT(*)                                                                AS receiver_tx_count,
            SUM(amount_received)                                                    AS receiver_total_amount,
            AVG(amount_received)                                                    AS receiver_avg_amount,
            MAX(amount_received)                                                    AS receiver_max_amount,
            COUNT(DISTINCT from_account)                                            AS receiver_distinct_senders,
            COUNT(DISTINCT payment_format)                                          AS receiver_distinct_formats,
            COUNT(DISTINCT receiving_currency)                                      AS receiver_distinct_currencies,
            ROUND(SUM(CASE WHEN amount_received % 1 = 0 THEN 1 ELSE 0 END)
                  * 100.0 / COUNT(*), 2)                                            AS receiver_pct_round_amounts
        FROM transactions
        GROUP BY to_account
    ),

    velocity AS (
        SELECT
            from_account                                                            AS account,
            AVG(tx_count_last_24h)                                                  AS sender_avg_velocity_24h
        FROM (
            SELECT
                from_account,
                COUNT(*) OVER (
                    PARTITION BY from_account
                    ORDER BY CAST(timestamp AS TIMESTAMP)
                    RANGE BETWEEN INTERVAL 1 DAY PRECEDING AND CURRENT ROW
                ) AS tx_count_last_24h
            FROM transactions
        )
        GROUP BY from_account
    )

    SELECT
        COALESCE(s.account, r.account)      AS account,
        s.sender_tx_count,
        s.sender_total_amount,
        s.sender_avg_amount,
        s.sender_max_amount,
        s.sender_distinct_receivers,
        s.sender_distinct_formats,
        s.sender_distinct_currencies,
        s.sender_pct_round_amounts,
        v.sender_avg_velocity_24h,
        r.receiver_tx_count,
        r.receiver_total_amount,
        r.receiver_avg_amount,
        r.receiver_max_amount,
        r.receiver_distinct_senders,
        r.receiver_distinct_formats,
        r.receiver_distinct_currencies,
        r.receiver_pct_round_amounts
    FROM sender s
    FULL OUTER JOIN receiver r  ON s.account = r.account
    LEFT JOIN velocity v        ON s.account = v.account
''')

print(con.execute("SELECT * FROM account_features LIMIT 5").df())
