import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="AfyaMed", page_icon="🩺")
st.title("🩺 AfyaMed - Your Everyday Health Companion")

st.markdown("Ask me any gynecology-related question. I'll give you a helpful, research-based response. Please note this is not a substitute for medical advice.")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
    model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")
    return model, tokenizer

model, tokenizer = load_model()
# Pre-filled curated Q&A responses
custom_answers = {
    "what are the signs of endometriosis": "Signs of endometriosis include pelvic pain, painful periods, pain during intercourse, infertility, and fatigue.",
    "causes of painful periods": "Painful periods can be caused by endometriosis, fibroids, pelvic inflammatory disease, or hormonal imbalances.",
    "what is the best contraceptive for someone with pcos": "The best contraceptive for PCOS often depends on individual needs, but many doctors recommend hormonal birth control like the pill or hormonal IUD to regulate periods and reduce symptoms.",
    "why do i have spotting between periods": "Spotting between periods can be caused by hormonal fluctuations, stress, ovulation, birth control, or underlying conditions like fibroids or infections.",
    "what causes vaginal itching": "Vaginal itching may be caused by yeast infections, bacterial vaginosis, allergic reactions, or sexually transmitted infections. It’s best to get checked if it persists.",
    "can hormonal birth control cause mood changes": "Yes, hormonal birth control can cause mood changes in some people due to changes in hormone levels. If it’s severe, consult a healthcare provider.",
    "how do i know if i have pcos": "Common signs of PCOS include irregular periods, weight gain, acne, excessive hair growth, and difficulty getting pregnant.",
    "what are the early signs of pregnancy": "Early signs of pregnancy include missed periods, nausea, tender breasts, fatigue, and frequent urination.",
    "how can i relieve cramps naturally": "You can relieve cramps with heating pads, gentle exercise, hydration, herbal teas, and pain relievers like ibuprofen.",
    "when should i see a gynecologist": "You should see a gynecologist if you experience unusual bleeding, pain, missed periods, or have questions about contraception or reproductive health.",
    "how do i know if i have an sti": "Common signs of an STI include unusual vaginal discharge, itching, burning during urination, lower abdominal pain, or sores around the genitals. Some STIs have no symptoms, so regular testing is important.",
    "can you get an sti from oral sex": "Yes, STIs like gonorrhea, herpes, and HPV can be spread through oral sex. It’s important to use protection even during oral contact.",
    "what is the difference between uti and sti": "A UTI affects the urinary tract and usually causes burning during urination and frequent urges to pee. STIs can cause similar symptoms but are passed through sexual contact and may involve discharge, pain, or sores.",
    "where can i get free sti testing in kenya": "You can get free STI testing at most public health facilities and youth-friendly centers like LVCT Health, Marie Stopes Kenya, and government hospitals.",
    "what causes pain during sex": "Pain during sex can be caused by dryness, infections, fibroids, or psychological stress. It’s important to speak to a gynecologist if it keeps happening.",
    "how often should i go for a pap smear": "In Kenya, women are advised to start cervical cancer screening at age 25 and repeat every 3 years if results are normal.",
    "why does my period last more than 7 days": "Prolonged periods can be caused by fibroids, hormonal imbalance, or infections. If it happens often, visit a clinic for testing.",
    "what are signs i need to see a gynecologist": "See a gynecologist if you experience unusual bleeding, persistent pelvic pain, abnormal discharge, missed periods, or pain during sex.",
    "what is the most used contraceptive in kenya": "The most commonly used contraceptives in Kenya are injectables, followed by implants and pills. The best option depends on your lifestyle and health history.",
    "can i use the morning after pill more than once": "Emergency pills can be used more than once but are not meant for regular use. If you use them often, consider a long-term contraceptive method.",
    "do contraceptives cause infertility": "Contraceptives do not cause infertility. Your fertility returns shortly after stopping most methods, though cycles may take time to normalize.",
    "where can i get free contraceptives in kenya": "Public hospitals, clinics, and organizations like Family Health Options Kenya and Marie Stopes provide free or low-cost contraceptives.", 
    "what are common stis in kenya": "The most common STIs in Kenya include gonorrhea, chlamydia, syphilis, trichomoniasis, herpes, and HIV. Many of these can be treated if diagnosed early.",
    "can i get an sti if my partner shows no symptoms": "Yes, many STIs can be transmitted even when your partner has no symptoms. Regular testing for both partners is important.",
    "do i need to tell my partner if i have an sti": "Yes, it's important to inform your partner if you test positive for an STI so they can get tested and treated too. It also helps prevent reinfection.",
    "can stis be cured": "Some STIs like gonorrhea, chlamydia, and syphilis can be cured with antibiotics. Others like herpes and HIV can be managed but not fully cured.",
    "what causes discharge with a bad smell": "Foul-smelling vaginal discharge may be caused by bacterial vaginosis, trichomoniasis, or other infections. Visit a health facility for testing and treatment.",
    "what is a normal menstrual cycle": "A normal menstrual cycle lasts between 21 to 35 days. Bleeding usually lasts 2 to 7 days. If yours is outside this range, talk to a gynecologist.",
    "why does my period stop then start again": "This can be due to hormonal fluctuations, stress, or birth control. If it happens often, it’s worth checking with a healthcare provider.",
    "can fibroids cause frequent urination": "Yes, fibroids pressing against the bladder can lead to frequent urination. A pelvic scan can help determine if fibroids are the cause.",
    "can i get pregnant while on birth control": "No method is 100% effective, but most contraceptives are highly reliable if used correctly. Missing doses or late injections increases risk.",
    "which is better, implant or injection": "Both are effective. Implants last 3–5 years and are low maintenance. Injections require regular clinic visits every 3 months. Choice depends on your lifestyle.",
    "what are the side effects of contraceptive pills": "Side effects can include nausea, weight changes, mood swings, and spotting. Most are temporary and go away after a few months.",
    "can i get family planning without parental consent": "Yes, in Kenya, people aged 18 and above can access contraceptives without parental consent. Some youth-friendly clinics support even younger teens confidentially.",
    "can untreated stis cause infertility": "Yes, untreated STIs like chlamydia and gonorrhea can lead to pelvic inflammatory disease (PID), which increases the risk of infertility.",
    "can i get reinfected with an sti after treatment": "Yes, if your partner isn’t treated too, you can get reinfected. Always complete treatment and encourage partners to test as well.",
    "how long should i wait to test after unprotected sex": "STIs like chlamydia and gonorrhea can be detected within 1–2 weeks. HIV may take 2–6 weeks. Visit a clinic for appropriate testing timelines.",
    "can stis go away on their own": "Most STIs do not go away without treatment. Leaving them untreated can lead to serious complications, so get tested and treated early.",
    "is brown discharge before periods normal": "Brown discharge can be old blood from a previous cycle. If it’s consistent, has a bad smell, or happens often, get checked.",
    "why do i get itchy after shaving down there": "Itching after shaving is usually due to irritation or ingrown hairs. Use clean tools and shave in the direction of hair growth.",
    "what is pelvic inflammatory disease": "Pelvic Inflammatory Disease (PID) is an infection of the reproductive organs, usually caused by untreated STIs. It can cause pain and fertility issues.",
    "does stress affect your period": "Yes, stress can affect your hormones and delay or stop your period. If it keeps happening, consult a gynecologist to rule out other causes.",
    "what happens if i miss a contraceptive pill": "If you miss one pill, take it as soon as you remember. If you miss two or more, follow the leaflet instructions or consult a clinic.",
    "how long after stopping the injection can i get pregnant": "Fertility may take several months to return after stopping the injection (Depo-Provera). Some women take up to a year to conceive.",
    "can i get contraceptives if i’m breastfeeding": "Yes, there are contraceptives safe for breastfeeding, like the mini-pill, implant, or IUD. Avoid estrogen-based methods in the first 6 months.",
    "can family planning mess up my periods": "Yes, some methods like implants or injections may cause irregular bleeding or stop periods. These effects are common and usually not harmful."

}
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Enter your health related question...")

import re

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Normalize input for custom Q&A
    import re
    key = re.sub(r"[^\w\s]", "", user_input.strip().lower())

    # Start with curated response if available
    if key in custom_answers:
        answer = custom_answers[key]
    else:
        # Fall back to BioGPT if not in curated list
        smart_prompt = f"{user_input.strip()}"
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=100)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Now check if BioGPT gave a useless answer
        normalized_answer = re.sub(r"[^\w\s]", "", answer.strip().lower())
        normalized_input = re.sub(r"[^\w\s]", "", user_input.strip().lower())

        if not answer.strip() or len(answer.strip().split()) < 10:
            answer = "I'm not sure how to answer that yet, but I'm learning! Try rephrasing or check with a doctor."

    with st.chat_message("assistant"):
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})




        
    
        
