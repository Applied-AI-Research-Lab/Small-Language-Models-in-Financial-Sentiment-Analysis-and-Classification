# Round-2 C1 results

```
                       system   n  accuracy  macro_f1  weighted_f1  acc_unanimous  acc_split  parse_failures  mean_time_sec
     qwen framework (Round 1) 340     94.41    0.9348       0.9430          96.13      76.67             NaN            NaN
 qwen sentence-only (Round 1) 340     90.88    0.8997       0.9102          90.97      90.00             NaN            NaN
                     qwen cot 340     81.47    0.8282       0.8186          80.97      86.67             0.0          12.75
                     qwen knn 340     95.59    0.9492       0.9562          96.45      86.67             0.0           6.36
           qwen supericl_bert 340     92.35    0.9208       0.9241          93.87      76.67             0.0           7.46
            qwen supericl_svm 340     93.82    0.9335       0.9387          95.16      80.00             0.0           7.04
    gemma framework (Round 1) 340     91.76    0.9164       0.9194          93.23      76.67             NaN            NaN
gemma sentence-only (Round 1) 340     79.41    0.8119       0.7993          79.35      80.00             NaN            NaN
                    gemma cot 340     71.18    0.7201       0.7100          70.65      76.67             0.0          12.87
                    gemma knn 340     75.59    0.7903       0.7625          75.48      76.67             0.0           7.67
          gemma supericl_bert 340     80.88    0.8320       0.8167          82.58      63.33             0.0           8.94
           gemma supericl_svm 340     78.82    0.8009       0.7960          80.97      56.67             0.0           8.90
```

## Exact McNemar (job = A)

| job | n | vs framework (A✓B✗ / A✗B✓, p) | vs sentence-only (A✓B✗ / A✗B✓, p) |
|---|---|---|---|
| qwen/cot | 340 | 16/60, 0.0000 | 11/43, 0.0000 |
| qwen/knn | 340 | 14/10, 0.5413 | 22/6, 0.0037 |
| qwen/supericl_bert | 340 | 12/19, 0.2810 | 24/19, 0.5424 |
| qwen/supericl_svm | 340 | 13/15, 0.8506 | 24/14, 0.1433 |
| gemma/cot | 340 | 11/81, 0.0000 | 21/49, 0.0011 |
| gemma/knn | 340 | 10/65, 0.0000 | 14/27, 0.0596 |
| gemma/supericl_bert | 340 | 5/42, 0.0000 | 34/29, 0.6147 |
| gemma/supericl_svm | 340 | 7/51, 0.0000 | 32/34, 0.9022 |
