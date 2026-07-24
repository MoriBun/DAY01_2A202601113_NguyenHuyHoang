# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng đánh dấu "Câu trả lời của bạn" bằng
câu trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Ở temperature = 0.0, model gần như luôn chọn câu trả lời có xác suất cao nhất kiểu 1 fact rất phổ biến về Việt Nam, chạy nhiều lần thì kết quả sẽ gần như nhau. Khi tăng lên 0.5, 1.0, 1.5, model càng ngày có các câu trả lời có các chủ đề và cách diễn đạt trở nên đa dạng hơn. Tóm lại, temperature càng cao thì output càng đa dạng tuy vậy có thể thiếu nhất quán và kém chính xác hơn. 

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Em nghĩ nên đặt temperature khá thấp, khoảng 0.0-0.3, cho chatbot hỗ trợ khách hàng. Vì loại Chatbot này cần trả lời nhất quán, chính xác và bám sát thông tin sản phẩm, không nên sáng tạo hay đưa ra thông tin khác nhau cho cùng một câu hỏi vì điều đó dễ gây hiểu lầm và mất niềm tin của khách hàng.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> 10.000 người x 3 lần/ngày x 350 token = 10.500.000 token/ngày. Từ đó ta có chi phí sẽ là GPT-4o = 10.500 x 0.01 đô = 105 đô/ngày còn GPT-4-o-mini = 10.500 x 0.0006 = 6.3 đô/ngày. Tức là GPT-4o đắt hơn 16.7 lần. GPT-4o đáng được dùng trong các tác vụ đồi hỏi suy luận phức tạp, độ chính xác cao hoặc ảnh hưởng đến trải nghiệm/ uy tín sản phẩm ví dụ tư vấn pháp lý/ y tế hoặc xử lý yêu cầu kĩ thuật phức tạp. Ngược lại, GPT-4o-mini phù hợp với các tác vụ đơn giản, khối lượng lớn như trả lời FAQ, phân loại intent, hay tóm tắt ngắn, nơi mà tốc độ và chi phí quan trọng hơn độ thông minh của model.
---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Hai phản hồi khác nhau rõ rệt về độ dài, từ vựng lẫn cách trình bày. Với persona giáo dục tiểu học thì model trả lời ngắn gọn, tránh các thuật ngữ chuyên ngành. Ngược lại với persona tài chính, model trả lời dài hơn và trình bày có cấu trúc, các từ ngữ kĩ thuật.Tóm gọn lại là điều này cho thấy system prompt không chỉ định hướng "giọng văn" mà còn ảnh hưởng trực tiếp đến độ dài, mức độ chuyên sâu và cấu trúc trình bày, dù cùng 1 câu hỏi và cùng model.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
>   Chênh nhau khoảng 5.4%. Điều này cho thấy công thức 0.75 từ ≈ 1 token vốn chỉ là quy tắc kinh nghiệm hiệu chỉnh theo tiếng Anh, nên áp cho tiếng Việt không hoàn toàn chính xác — có thể lệch theo cả hai chiều tùy đoạn văn cụ thể. Về bản chất, tiếng Việt thường tốn nhiều token hơn tiếng Anh cùng độ dài (cùng số ký tự/ý nghĩa) vì bộ mã hóa BPE của các model như GPT được huấn luyện chủ yếu trên dữ liệu tiếng Anh — các từ tiếng Anh phổ biến thường được gộp thành 1 token duy nhất, trong khi các ký tự có dấu của tiếng Việt (ă, â, ê, ô, ơ, ư và các dấu thanh) là chuỗi byte hiếm gặp hơn trong dữ liệu huấn luyện, nên thường bị tách thành nhiều token nhỏ hơn cho cùng một từ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng tương tác trực tiếp với người dùng như chatbot hay trợ lý ảo, đặc biệt khi câu trả lời dài — vì nó giúp người dùng thấy nội dung xuất hiện ngay lập tức thay vì phải chờ cả câu trả lời sinh xong, tạo cảm giác phản hồi nhanh và giảm cảm giác "đứng hình" dù tổng thời gian xử lý không đổi. Ngược lại, non-streaming phù hợp hơn khi hệ thống cần dùng toàn bộ phản hồi trước khi xử lý tiếp — ví dụ khi output cần parse thành JSON, validate, hoặc làm input cho bước tự động hóa tiếp theo trong pipeline — vì lúc đó không có người xem trực tiếp nên việc stream từng phần không mang lại lợi ích, chỉ làm code phức tạp hơn. 

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> So với delay cố định, exponential backoff giúp giãn thời gian giữa các lần retry ngày càng xa nhau, giảm áp lực dồn dập lên server đang quá tải và cho nó thời gian hồi phục thay vì tiếp tục nhận request với cùng tần suất ban đầu. Nếu hàng nghìn client cùng dùng delay cố định (ví dụ luôn chờ đúng 1 giây), tất cả sẽ đồng loạt gửi lại request cùng lúc, tạo thành từng đợt sóng request lặp lại đều đặn ("thundering herd") khiến server vốn đã quá tải càng khó hồi phục, thậm chí có thể sập hẳn. Exponential backoff (thường kết hợp thêm jitter — độ trễ ngẫu nhiên nhỏ) giúp phân tán các lượt retry ra theo thời gian, tránh hiện tượng đồng bộ hóa này.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System prompt: "Bạn là một trợ lý học tập thân thiện, chuyên hỗ trợ sinh viên ôn tập kiến thức lập trình Python và AI cơ bản. Luôn trả lời ngắn gọn, dễ hiểu, ưu tiên ví dụ minh họa thực tế, và trả lời bằng tiếng Việt trừ khi người dùng chủ động hỏi bằng tiếng Anh." Hai lựa chọn từ ngữ quan trọng: (1) "trả lời ngắn gọn" — vì trợ lý chạy dạng CLI với streaming, câu trả lời dài vừa khiến người dùng chờ lâu hơn để đọc hết, vừa tốn nhiều token output hơn (tăng chi phí mỗi lượt); (2) "trả lời bằng tiếng Việt trừ khi..." — chỉ định rõ ngôn ngữ mặc định giúp trợ lý nhất quán với đối tượng người dùng chính, tránh trường hợp model tự ý chuyển sang tiếng Anh khi câu hỏi có lẫn từ tiếng Anh.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất hiện tại là history chỉ giữ 3 lượt hội thoại gần nhất (6 message) và không lưu trữ gì sau khi chương trình kết thúc — nên trợ lý vừa "quên" ngữ cảnh xa hơn trong một phiên dài, vừa mất hoàn toàn ngữ cảnh nếu người dùng thoát rồi mở lại. Ngoài ra, hệ thống cũng chưa có bước kiểm duyệt nội dung đầu vào/đầu ra. Cải thiện cụ thể: thay vì cắt bỏ hoàn toàn các lượt cũ khi vượt quá giới hạn, có thể thêm một bước tóm tắt (summarization) — khi history sắp vượt 3 lượt, gọi model tóm tắt các lượt cũ nhất thành 1-2 câu ngắn, lưu vào một message ở đầu history thay cho toàn bộ nội dung gốc. Cách này giữ được ngữ cảnh dài hạn quan trọng mà không làm tăng vô hạn số token input mỗi lượt gọi.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
