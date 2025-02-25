# 🎧 플레이리스트 추천
## 개요
> 자연어 처리 모델을 활용해 플레이리스트 추천 모델 구축

## 목적
> 사용자 기분에 따른 플레이리스트 추천해주기

## 모델 학습 결과
|EPOCH|ITERVARL|
|---|---|
|50|100|

||Accuracy|Loss|
|---|---|---|
|TRAIN|0.8841|0.2349|
|TEST|0.7675|0.6342|

### 데이터셋
> 유튜브 댓글 크롤링

<details>
  <summary>우울 플레이리스트 데이터 셋</summary>
    <table border="1">
    <tr>
      <th>데이터 이름</th>
      <th>내용</th>
      <th>데이터 형식</th>
    </tr>
    <tr>
      <td>ㅣabel_0_1</td>
      <td>[우울 플레이리스트] 박제가 된 천재를 아시오?</td>
      <td>xlsx</td>
    </tr>
    <tr>
      <td>ㅣabel_0_2</td>
      <td>[우울 플레이리스트] 내게 여운을 주는 문학 작품</td>
      <td>xlsx</td>
    </tr>
    <tr>
      <td>ㅣabel_0_3</td>
      <td>[우울 플레이리스트] 남들에게 말 못할 비밀과 고민들</td>
      <td>xlsx</td>
    </tr>
    <tr>
      <td>ㅣabel_0_4</td>
      <td>[우울 플레이리스트] 유독 힘이 드는 날 지친 당신을 위한 우울한 팝송</td>
      <td>xlsx</td>
    </tr>
    <tr>
      <td>ㅣabel_0_5</td>
      <td>[우울 플레이리스트] 떠나려 하는 모든 이에게</td>
      <td>xlsx</td>
    </tr>
    <tr>
      <td>ㅣabel_0_6</td>
      <td>[우울 플레이리스트] 마음이 죽은 사람들에게, 가사 없는 음악</td>
      <td>xlsx</td>
    </tr>
    <tr>
      <td>ㅣabel_0_7</td>
      <td>[우울 플레이리스트] 잘 지내요? 난 그저 잘 지낼테니, 당신은 잘 지내야 해요</td>
      <td>xlsx</td>
    </tr>
    <tr>
      <td>ㅣabel_0_8</td>
      <td>[우울 플레이리스트] 그리워 말고 추억으로 남겨둬</td>
      <td>xlsx</td>
    </tr>
    <tr>
      <td>ㅣabel_0_9</td>
      <td>[우울 플레이리스트] 유독 우울한 밤에 듣기좋은 감성팝송</td>
      <td>xlsx</td>
    </tr>
    <tr>
      <td>ㅣabel_0_10</td>
      <td>[우울 플레이리스트] 사랑에는 유통기한이 없을거라 생각했다.</td>
      <td>xlsx</td>
    </tr>
    <tr>
      <td>ㅣabel_0</td>
      <td>[우울 플레이리스트] 부끄럼 많은 생애를 보냈습니다.</td>
      <td>xlsx</td>
    </tr>
  </table>
</details>


  
<details>
  <summary>해피 플레이리스트 데이터 셋</summary>
    <table border="1">
    <tr>
      <th>데이터 이름</th>
      <th>내용</th>
      <th>데이터 형식</th>
    </tr>
    <tr>
      <td>ㅣabel_1_1</td>
      <td>[행복 플레이리스트] 월요병 ㅃㅃ</td>
      <td>xlsx</td>
    </tr>
    <tr>
      <td>ㅣabel_1_2</td>
      <td>[행복 플레이리스트] 오래된 부모님의 연애시절 앨범을 꺼내보았다</td>
      <td>xlsx</td>
    </tr>
    <tr>
      <td>ㅣabel_1_3</td>
      <td>[행복 플레이리스트] 학창시절 나의 첫 연애 | 그 해 우리는</td>
      <td>xlsx</td>
    </tr>
    <tr>
      <td>ㅣabel_1_4</td>
      <td>[행복 플레이리스트] 살면서 가장 설렜던 순간속으로</td>
      <td>xlsx</td>
    </tr>
    <tr>
      <td>ㅣabel_1_5</td>
      <td>[행복 플레이리스트] 첫눈에 반했던 경험</td>
      <td>xlsx</td>
    </tr>
    <tr>
      <td>ㅣabel_1_6</td>
      <td>[행복 플레이리스트] 좋아하는 사람과 연애하면 하고 싶은 것</td>
      <td>xlsx</td>
    </tr>
    <tr>
      <td>ㅣabel_1_7</td>
      <td>[행복 플레이리스트] 벌써부터 크리스마스 기다리는 사람?</td>
      <td>xlsx</td>
    </tr>
    <tr>
      <td>ㅣabel_1_8</td>
      <td>[행복 플레이리스트] 행복을 살 수 있는 가게가 있나요?</td>
      <td>xlsx</td>
    </tr>
    <tr>
      <td>ㅣabel_1_9</td>
      <td>[행복 플레이리스트] 따사로운 봄을 기다리며</td>
      <td>xlsx</td>
    </tr>
    <tr>
      <td>ㅣabel_1</td>
      <td>[행복 플레이리스트] 첫 사랑 썰 푸는 곳</td>
      <td>xlsx</td>
    </tr>
  </table>
</details>

### 요구사항
* python == 3.8.2
* pandas == 2.0.3
* numpy == 1.24.3
* scikit-learn == 1.3.2
* pytorch == 2.3.0
* tabulate == 0.9.0
* matplotlib == 3.7.5
* seaborn == 0.13.2
* koreanize-matplotlib == 0.1.1
* joblib == 1.4.2
