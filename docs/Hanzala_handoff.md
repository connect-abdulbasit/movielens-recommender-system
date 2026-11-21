# Hanzala ka hissa

## Basit ke saath merge
ZIP mein complete working copy hai. Apne existing clone mein sirf assigned changes ke liye ZIP se handoff/hanzala-naive-bayes.patch project root mein copy karo. Basit ka latest code agar ZIP wale version ke barabar hai to repo root se chalao:

    git switch -c naive-bayes
    git apply --check hanzala-naive-bayes.patch
    git apply hanzala-naive-bayes.patch
    python -m unittest tests.test_naive_bayes
    python main.py recommend --user 196 --method nb
    git add src/naive_bayes.py src/metrics.py src/explain.py tests/test_naive_bayes.py
    git commit -m "Implement genre-based Naive Bayes with explanations"

Agar branch pehle se hai to git switch naive-bayes use karo. Patch lagane se pehle apni existing changes check karo. Patch ke andar tumhara model, metrics edge cases, NB explanation aur tests hain. Complete ZIP mein shared evaluator, baselines, results, README aur report bhi hain. Group merge ke waqt ye separately compare kar lena. ZIP ko chalane ke liye original ZIP se data/ml-100k/ isi project root mein copy karo.

## Viva ke liye
1. Matrix mein zero ka matlab movie abhi rate nahi hui. Model usko dislike count nahi karta.
2. Har user ke liye 4 ya 5 stars like hain. 1, 2 aur 3 dislike hain. Har genre ka flag 0 ya 1 hota hai.
3. Misaal: user ne ek action film like ki aur ek non-action film dislike ki. Alpha=1 se P(Action|Like)=2/3 aur P(Action|Dislike)=1/3. Prior odds 1:1 hon aur baqi flags dono classes mein barabar hon to action movie ke liye posterior odds 2:1, yani like ki probability 2/3 hai.
4. Model present aur absent dono flags ka evidence use karta hai. Logs add karte hain taake bohat choti probabilities multiply karne ka masla na ho. Output mein present genres ka asar bataya jata hai.
5. Training mein rated movies hata kar baqi ko probability ke order mein rakhte hain. Same flags wali movies ka score equal hota hai. Tie mein chhoti movie ID pehle aati hai.
6. Agar top 10 mein 2 relevant hits hain aur user ne test mein 5 movies like ki hain to precision 2/10 aur recall 2/5 hai. RMSE aur MAE star prediction ki ghalti naapte hain. NB ki 0.84 probability ka matlab 84% estimated chance of a like hai.
