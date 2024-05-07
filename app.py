import streamlit as st
from rag import rag_

# Tạo ra một dictionary lưu trữ mapping giữa câu hỏi và giá trị tương ứng của nút
guiding_questions = {
    "Thời gian đào tạo của chương trình thạc sĩ?": False,
    "Điều kiện được bảo vệ luận văn thạc sĩ?": False,
    "Tiêu chuẩn huy chương vàng được quy định như thế nào?": False,
    "Miễn thi được quy định như thế nào?": False,
    "Điểm I là điểm gì?": False
}

def main():
    st.set_page_config(page_title="Chatbot BK")

    st.image(["logo.jpg"], width=100)
    # with st.columns(3)[1]:
    #     st.image(["logo.jpg"])

    st.title("Chatbot Phòng Đào Tạo")
    # st.markdown("<h1 style='text-align: center'>Chatbot Phòng Đào Tạo</h1>", unsafe_allow_html=True)
    
    st.subheader("Tôi có thể giải đáp các thắc mắc về quy định học vụ của Trường Đại Học Bách Khoa - ĐHQG TP.HCM", divider='rainbow')

    # Hiển thị sidebar với các câu hỏi hướng dẫn
    st.sidebar.subheader("Có thể bạn quan tâm những câu hỏi dưới đây:")

    # set initial message
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Xin chào, tôi có thể giúp gì cho bạn"}
        ]

    if "messages" in st.session_state.keys():
        # display messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # get user input
    user_prompt = st.chat_input()
    for question in guiding_questions.keys():
        if st.sidebar.button(question, key=question):
            user_prompt = question
            guiding_questions[question] = True  # Đánh dấu câu hỏi được chọn
    handle_user_input1(user_prompt)

def handle_user_input(user_prompt):
    if user_prompt is not None:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

    # process user input
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Loading..."):
                ai_response = rag_(user_prompt)
                if ai_response == "Encountered some errors. Please recheck your request!":
                    st.write("Xin lỗi, tôi không có thông tin về câu hỏi này!")
                else:
                    st.write(ai_response)

        new_ai_message = {"role": "assistant", "content": ai_response}
        st.session_state.messages.append(new_ai_message)


def handle_user_input1(user_prompt):
    # Clear all messages except the latest user and assistant messages
    st.session_state.messages = [msg for msg in st.session_state.messages if msg["role"] in ["user", "assistant"]]

    if user_prompt is not None:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

    # process user input
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.spinner("Loading..."):
            ai_response = rag_(user_prompt)
            if ai_response == "Encountered some errors. Please recheck your request!":
                st.session_state.messages.append({"role": "assistant", "content": "Xin lỗi, tôi không có thông tin về câu hỏi này!"})
            else:
                st.session_state.messages.append({"role": "assistant", "content": ai_response})

    # Display only the latest user and assistant messages
    latest_user_message = next((msg for msg in reversed(st.session_state.messages) if msg["role"] == "user"), None)
    latest_assistant_message = next((msg for msg in reversed(st.session_state.messages) if msg["role"] == "assistant"), None)
    
    if latest_user_message:
        with st.chat_message("user"):
            st.write(latest_user_message["content"])

    if latest_assistant_message:
        with st.chat_message("assistant"):
            st.write(latest_assistant_message["content"])



if __name__ == '__main__':
    main()
