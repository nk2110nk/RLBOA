import os

# 'value'ディレクトリのパス
directory_path = 'det=False_noise=False'

# 結果を格納するリスト
results = []

# 'value'ディレクトリ内のすべてのファイルを確認
for filename in sorted(os.listdir(directory_path)):
    file_path = os.path.join(directory_path, filename)

    # ファイルが実際にファイルであることを確認
    if os.path.isfile(file_path):
        try:
            # 各列の数値を格納するリスト
            column_sums = [0] * 6  # 6列の合計を格納
            row_count = 0  # 有効な行数をカウント

            # ファイルを読み込む
            with open(file_path, 'r') as file:
                for line in file:
                    # 1行をスペースで区切って数字に変換
                    numbers = list(map(float, line.split()))
                    if len(numbers) == 6:  # 6つの数があるか確認
                        # 各列の合計を更新
                        for i in range(6):
                            column_sums[i] += numbers[i]
                        row_count += 1

            # 行数が1以上なら平均を計算
            if row_count > 0:
                column_averages = [s / row_count for s in column_sums]
            else:
                column_averages = [0] * 6

            # 結果をリストに格納（ファイル名を除く）
            results.append('\t'.join(map(str, column_averages)))

        except Exception as e:
            print(f"ファイル {filename} を処理中にエラーが発生しました: {e}")

# 出力ファイルに書き込む
output_file = 'column_averages.txt'
with open(output_file, 'w') as output:
    for result in results:
        output.write(result + '\n')

print(f"平均値を {output_file} に出力しました。")
