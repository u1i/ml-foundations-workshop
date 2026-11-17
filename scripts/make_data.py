"""Version 3 (17 Nov 2026): Arabic requests also in dialect (Gulf, Levantine, Egyptian), variants supplied by the Arabic-speaking reviewer.
Version 2 (15 Nov 2026): column names and categories updated after review.

Synthetic datasets for the Practical ML & Data Science Foundations workshop.

All data is synthetic. Names of services and units are generic. Run:
    python scripts/make_data.py
Writes CSV files into data/. Fixed seeds, so every run produces the same files.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------- 1. visa applications (Day 1)
def visa_applications(n=5000, seed=1):
    rng = np.random.default_rng(seed)
    visa_type = rng.choice(["Tourist", "Employment", "Family", "Student"], n, p=[0.40, 0.30, 0.20, 0.10])
    region = rng.choice(["GCC", "Asia", "Europe", "Africa", "Americas"], n, p=[0.15, 0.40, 0.20, 0.15, 0.10])
    channel = rng.choice(["Online", "Service centre", "Mobile app"], n, p=[0.50, 0.25, 0.25])
    # sponsor follows the visa type
    sponsor = np.select(
        [visa_type == "Employment", visa_type == "Student", visa_type == "Family"],
        ["Employer", "Educational Institution", "Individual"],
        default=rng.choice(["Tourism Establishment", "Individual", "Not Applicable"], n, p=[0.55, 0.30, 0.15]))
    age = np.clip(rng.normal(34, 10, n), 18, 75).round().astype(int)
    prev_visits = rng.poisson(1.2, n)
    docs_complete = rng.random(n) < 0.80
    security = rng.random(n) < np.where(np.isin(visa_type, ["Employment", "Student"]), 0.25, 0.10)
    doc_count = np.clip(rng.normal(5, 2, n), 1, 12).round().astype(int)
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")

    base = pd.Series(visa_type).map({"Tourist": 4, "Employment": 9, "Family": 8, "Student": 7}).to_numpy()
    days = (base
            + np.where(docs_complete, 0, 6)
            + np.where(security, 7, 0)
            + pd.Series(channel).map({"Online": 0, "Service centre": 1.5, "Mobile app": -0.5}).to_numpy()
            - np.minimum(prev_visits, 4) * 0.6
            + pd.Series(sponsor).map({"Individual": 1, "Employer": 0, "Tourism Establishment": -1, "Educational Institution": -1, "Not Applicable": 0}).to_numpy()
            + 0.3 * (doc_count - 5)
            + rng.normal(0, 2.0, n))
    days = np.clip(days, 1, None).round().astype(int)

    df = pd.DataFrame({
        "application_id": [f"VA-{100000 + i}" for i in range(n)],
        "submission_date": dates.strftime("%Y-%m-%d"),
        "visa_type": visa_type, "applicant_region": region, "channel": channel, "sponsor_type": sponsor,
        "applicant_age": age.astype(object), "previous_visits": prev_visits,
        "documents_complete": np.where(docs_complete, "Yes", "No").astype(object),
        "additional_document_review_required": np.where(security, "Yes", "No"), "supporting_document_count": doc_count,
        "processing_days": days,
    })
    clean = df.copy()
    clean["applicant_age"] = clean["applicant_age"].astype(int)

    # --- make the raw version messy, the way real exports are
    raw = df.copy()
    r = np.random.default_rng(seed + 100)
    # the same category written in different ways, including Arabic
    variants = {"Tourist": ["tourist", "TOURIST", "سياحة", "Tourist "], "Employment": ["employment", "EMPLOYMENT", "Work", "عمل", "شغل"],
                "Family": ["family", "Family visit", "عائلية"], "Student": ["student", "دراسة", "طالب"]}
    idx = r.choice(n, int(n * 0.08), replace=False)
    raw.loc[idx, "visa_type"] = [r.choice(variants[v]) for v in raw.loc[idx, "visa_type"]]
    idx = r.choice(n, int(n * 0.05), replace=False)
    raw.loc[idx, "documents_complete"] = r.choice(["Y", "N", "yes", "no", "1", "0"], len(idx))
    # missing values
    raw.loc[r.choice(n, int(n * 0.06), replace=False), "applicant_age"] = np.nan
    raw.loc[r.choice(n, int(n * 0.03), replace=False), "channel"] = np.nan
    raw.loc[r.choice(n, int(n * 0.02), replace=False), "supporting_document_count"] = np.nan
    # wrong types: age as text with units, and impossible values
    idx = r.choice(n, 60, replace=False)
    raw.loc[idx, "applicant_age"] = [f"{a} years" if pd.notna(a) else a for a in raw.loc[idx, "applicant_age"]]
    raw.loc[r.choice(n, 12, replace=False), "applicant_age"] = 999
    # outliers in the target: files that were put on hold
    idx = r.choice(n, 15, replace=False)
    raw.loc[idx, "processing_days"] = r.integers(90, 160, len(idx))
    # a different date format from one service centre
    idx = r.choice(n, 150, replace=False)
    raw.loc[idx, "submission_date"] = pd.to_datetime(raw.loc[idx, "submission_date"]).dt.strftime("%d/%m/%Y")
    # a few exact duplicates
    raw = pd.concat([raw, raw.sample(25, random_state=seed)], ignore_index=True)
    raw = raw.sample(frac=1, random_state=seed).reset_index(drop=True)
    return raw, clean


# ---------------------------------------------------------------- 2. tax registrations (Day 1)
def tax_registrations(n=4000, seed=2):
    rng = np.random.default_rng(seed)
    activity = rng.choice(["Trading", "Services", "Manufacturing", "Consulting", "Real estate", "Food and beverage"], n)
    legal = rng.choice(["Sole establishment", "LLC", "Free zone company", "Branch of foreign company"], n, p=[0.30, 0.40, 0.20, 0.10])
    years = np.clip(rng.exponential(4, n), 0, 30).round(1)
    turnover = np.round(np.exp(rng.normal(13.5, 1.0, n)), -3)
    docs_missing = rng.poisson(0.6, n)
    licence_valid = rng.random(n) < 0.92
    online = rng.random(n) < 0.45
    mismatch = rng.random(n) < 0.12  # name on licence differs from application
    logit = (-3.25 + 1.6 * docs_missing + 3.5 * (~licence_valid) - 0.8 * online + 2.8 * mismatch
             - 0.08 * years + np.where(legal == "Branch of foreign company", 1.0, 0) + rng.normal(0, 0.3, n))
    rejected = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({
        "registration_id": [f"TR-{200000 + i}" for i in range(n)], "business_activity": activity, "legal_form": legal,
        "years_in_operation": years, "declared_turnover_aed": turnover.astype(int), "documents_missing": docs_missing,
        "trade_licence_valid": np.where(licence_valid, "Yes", "No"), "submitted_via_online_portal": np.where(online, "Yes", "No"), "name_mismatch": np.where(mismatch, "Yes", "No"),
        "rejected": rejected.astype(int)})


# ---------------------------------------------------------------- 3. benefit claims, about 2% fraud (Day 1, mini-project)
def benefit_claims(weeks=20, per_week=500, seed=3):
    rng = np.random.default_rng(seed)
    n = weeks * per_week
    week = np.repeat(np.arange(1, weeks + 1), per_week)
    program = rng.choice(["Social support", "Zakat assistance"], n, p=[0.65, 0.35])
    ctype = rng.choice(["General Support", "Housing-related Support", "Education-related Support"], n, p=[0.45, 0.35, 0.20])
    household = np.clip(rng.poisson(4, n), 1, 12)
    amount = np.round(np.exp(rng.normal(8.0, 0.5, n)), -1)
    claims_12m = rng.poisson(2, n)
    days_reg = rng.integers(5, 3000, n)
    bank_changes = rng.poisson(0.15, n)
    addr_changes = rng.poisson(0.2, n)
    employer_declared = rng.random(n) < 0.6
    hour = rng.integers(0, 24, n)
    same_device = rng.poisson(0.1, n)
    logit = (-7.2 + 2.2 * bank_changes + 1.1 * addr_changes + 1.8 * same_device + 0.0005 * (amount - 3000)
             + 2.0 * (days_reg < 60) + 1.0 * ((hour < 5) | (hour > 22)) + 0.3 * claims_12m + rng.normal(0, 0.3, n))
    fraud = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({
        "claim_id": [f"BC-{300000 + i}" for i in range(n)], "week": week, "program_type": program, "claim_type": ctype, "household_size": household,
        "claim_amount_aed": amount.astype(int), "claims_last_12_months": claims_12m, "days_since_registration": days_reg,
        "payment_details_update_count": bank_changes, "address_changes_12m": addr_changes,
        "income_information_declared": np.where(employer_declared, "Yes", "No"), "submission_hour": hour,
        "claims_from_same_device": same_device, "fraud": fraud.astype(int)})


# ---------------------------------------------------------------- 4. patients by service use (Day 1)
def patients(n=3000, seed=4):
    rng = np.random.default_rng(seed)
    groups = {  # share, age, gp, specialist, emergency, prescriptions, missed, telehealth, lab, chronic, distance
        "occasional": (0.40, 32, 1.5, 0.3, 0.2, 1, 0.3, 0.5, 1, 0.1, 8),
        "chronic, well managed": (0.25, 55, 6, 4, 0.3, 14, 0.4, 2, 8, 2.0, 10),
        "frequent emergency": (0.15, 40, 1, 0.5, 4.5, 4, 2.5, 0.2, 3, 0.8, 15),
        "digital first": (0.20, 29, 0.8, 0.5, 0.2, 2, 0.2, 7, 2, 0.3, 20),
    }
    rows = []
    names = list(groups)
    labels = rng.choice(len(names), n, p=[g[0] for g in groups.values()])
    for i, k in enumerate(labels):
        _, age, gp, sp, er, rx, miss, tele, lab, chron, dist = groups[names[k]]
        rows.append({
            "patient_id": f"PT-{400000 + i}",
            "age": int(np.clip(rng.normal(age, 8), 18, 90)),
            "gp_visits_12m": rng.poisson(gp), "specialist_visits_12m": rng.poisson(sp),
            "emergency_visits_12m": rng.poisson(er), "prescriptions_12m": rng.poisson(rx),
            "missed_appointments_12m": rng.poisson(miss), "telehealth_sessions_12m": rng.poisson(tele),
            "lab_tests_12m": rng.poisson(lab), "chronic_conditions": rng.poisson(chron),
            "distance_to_clinic_km": round(float(np.clip(rng.normal(dist, 5), 0.5, 60)), 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 5. late applications with one leaking input (Day 2, mini-project)
def late_applications(n=6000, seed=5):
    rng = np.random.default_rng(seed)
    service = rng.choice(["Residency renewal", "New residency", "Family sponsorship", "Visit visa"], n, p=[0.35, 0.25, 0.20, 0.20])
    channel = rng.choice(["Online", "Service centre", "Mobile app"], n, p=[0.50, 0.25, 0.25])
    applicants = rng.integers(0, 3800, n)  # some applicants apply more than once
    submitted = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D") + pd.to_timedelta(rng.integers(7, 20, n), unit="h")
    docs_uploaded = submitted - pd.to_timedelta(rng.exponential(3, n).round(1), unit="D")
    num_docs = rng.integers(2, 9, n)
    docs_complete = rng.random(n) < 0.78
    medical = np.isin(service, ["New residency", "Family sponsorship"]) & (rng.random(n) < 0.8)
    security = rng.random(n) < 0.15
    queue = rng.integers(20, 400, n)  # applications waiting at the time of submission
    weekend_or_thu = pd.Series(submitted).dt.day_of_week.isin([3, 4]).to_numpy()
    base = pd.Series(service).map({"Residency renewal": 2, "New residency": 6, "Family sponsorship": 7, "Visit visa": 1}).to_numpy()
    upload_gap = (submitted - docs_uploaded).total_seconds().to_numpy() / 86400
    days = (base + np.where(docs_complete, 0, 5) + np.where(medical, 2.5, 0) + np.where(security, 6, 0)
            + queue / 80 + np.where(weekend_or_thu, 2.0, 0) + np.where(upload_gap > 7, 2.5, 0) + rng.normal(0, 1.8, n))
    days = np.clip(days, 1, None).round().astype(int)
    final_approval = submitted.normalize() + pd.to_timedelta(days, unit="D")
    df = pd.DataFrame({
        "application_id": [f"LA-{500000 + i}" for i in range(n)], "applicant_id": [f"AP-{a:05d}" for a in applicants],
        "service": service, "channel": channel,
        "documents_uploaded_at": docs_uploaded.strftime("%Y-%m-%d %H:%M"), "submitted_at": submitted.strftime("%Y-%m-%d %H:%M"),
        "number_of_documents": num_docs, "documents_complete": np.where(docs_complete, "Yes", "No"),
        "medical_fitness_document_required": np.where(medical, "Yes", "No"),
        "additional_document_review_required": np.where(security, "Yes", "No"),
        "queue_at_submission": queue,
        "final_approval_date": final_approval.strftime("%Y-%m-%d"),  # only known after the decision: leaks
        "late": (days > 10).astype(int)})
    return df.sort_values("submitted_at").reset_index(drop=True)


# ---------------------------------------------------------------- 6. citizen requests in Arabic and English (mini-project)
def citizen_requests(n=1500, seed=6):
    rng = np.random.default_rng(seed)
    T = {
        "Licensing": {
            "en": ["I want to renew my trade licence but the system shows an error.", "How long does it take to issue a new trade licence?",
                   "Which documents do I need to add an activity to my licence?", "My licence expired last week, can I still renew it online?",
                   "I need to change the company name on my licence."],
            "ar": ["أرغب في تجديد الرخصة التجارية لكن النظام يظهر خطأ.", "كم تستغرق مدة إصدار رخصة تجارية جديدة؟",
                   "ما هي المستندات المطلوبة لإضافة نشاط إلى الرخصة؟", "انتهت صلاحية رخصتي الأسبوع الماضي، هل يمكنني تجديدها عبر الإنترنت؟",
                   "أحتاج إلى تغيير اسم الشركة في الرخصة."]},
        "Payments": {
            "en": ["I paid the fee twice, please refund one payment.", "The payment page does not accept my card.",
                   "I did not receive a receipt for my payment.", "Why was I charged a late fee?", "Can I pay the fees in instalments?"],
            "ar": ["دفعت الرسوم مرتين، يرجى استرداد إحدى الدفعتين.", "صفحة الدفع لا تقبل بطاقتي.",
                   "لم أستلم إيصالاً بالدفع.", "لماذا تم فرض غرامة تأخير علي؟", "هل يمكنني دفع الرسوم على أقساط؟"]},
        "Visa and residency": {
            "en": ["My residency renewal has been pending for three weeks.", "How can I sponsor my parents?",
                   "Is a medical test required for a residency renewal?", "My visa application was rejected, what can I do?",
                   "Can I extend my visit visa for another month?"],
            "ar": ["طلب تجديد الإقامة معلق منذ ثلاثة أسابيع.", "كيف يمكنني كفالة والديّ؟", "هل الفحص الطبي مطلوب لتجديد الإقامة؟",
                   "تم رفض طلب التأشيرة، ماذا يمكنني أن أفعل؟", "هل يمكنني تمديد تأشيرة الزيارة لشهر آخر؟"]},
        "Technical support": {
            "en": ["I cannot log in to the portal using UAE PASS.", "The app crashes when I upload a document.",
                   "I forgot my password and the reset link is not reaching me.", "The website is very slow today.", "The form does not let me submit."],
            "ar": ["لا يمكنني تسجيل الدخول إلى البوابة باستخدام الهوية الرقمية.", "التطبيق يتوقف عند رفع مستند.",
                   "نسيت كلمة المرور ورابط إعادة التعيين لا يصل.", "الموقع بطيء جداً اليوم.", "الاستمارة لا تسمح لي بالتقديم."]},
        "Complaints": {
            "en": ["The staff at the service centre were not helpful.", "I waited two hours and nobody assisted me.",
                   "Nobody answers the call centre.", "I want to file a complaint about the long waiting time.",
                   "I was given wrong information at the counter."],
            "ar": ["الموظفون في مركز الخدمة لم يكونوا متعاونين.", "انتظرت ساعتين ولم يخدمني أحد.", "لا أحد يرد على مركز الاتصال.",
                   "أرغب في تقديم شكوى بشأن طول وقت الانتظار.", "حصلت على معلومات خاطئة عند الكاونتر."]},
    }
    # dialect versions of each Arabic request (key: the Modern Standard Arabic sentence without its final full stop)
    AR_DIALECTS = {
        "أرغب في تجديد الرخصة التجارية لكن النظام يظهر خطأ": [
                "بدي جدد الرخصة التجارية بس النظام يعطيني خطأ",
                "عاوز اجدد الرخصة التجارية بي النظام في خطأ",
                "ابا اجدد الرخصة التجاريه بس النظام يعطيني خطأ"
        ],
        "كم تستغرق مدة إصدار رخصة تجارية جديدة؟": [
                "كم ياخذ وقت لاصدار رخصة تجارية جديده",
                "قد ايه ياخذ وقت لاصدار على رخصة جديده",
                "كم ياخذ وقت لاصدار رخصة تجارية يديده"
        ],
        "ما هي المستندات المطلوبة لإضافة نشاط إلى الرخصة؟": [
                "شو هي المستندات المطلوبه عشان نضيف نشاط للرخصة",
                "ايه المستندات المطلوبة عشان نضيف نشاط للرخصة؟",
                "شو هي الاوراق المطلوبة لاضافة نشاط؟"
        ],
        "انتهت صلاحية رخصتي الأسبوع الماضي، هل يمكنني تجديدها عبر الإنترنت؟": [
                "رخصتي خلصت الاسبوع الماضي بقدر جددها اونلاين؟",
                "الرخصة بتاعتي انتهت الاسبوع الفات بقدر اجددها اونلاين؟",
                "انتهت صلاحية رخصتي الأسبوع الماضي كيف ممكن اجددها اون لاين؟"
        ],
        "أحتاج إلى تغيير اسم الشركة في الرخصة": [
                "بدي غير اسم الشركة في الرخصة",
                "عايز اغير اسم الشركة في الرخصة",
                "ابا اغير اسم الشركة"
        ],
        "دفعت الرسوم مرتين، يرجى استرداد إحدى الدفعتين": [
                "دفعت الرسوم مرتين بالغلط بدي استرجع وحده منهم",
                "دفعت الرسوم مرتين بالغلط ابا استرجع واحده منهم",
                "دفعت الرسوم مرتين بالغلط ابا استرجع وحده من الدفعات"
        ],
        "صفحة الدفع لا تقبل بطاقتي": [
                "صفحة الدفع ما تقبل بطاقتي",
                "صفحة الدفع ماعم تقبل بطاقتي",
                "صفحة الدفع ماتروم تقبل بطاقتي"
        ],
        "لم أستلم إيصالاً بالدفع": [
                "ما استلمت ايصال الدفع",
                "ما استلمتش الفاتورة",
                "ما استلمت الفاتوره"
        ],
        "لماذا تم فرض غرامة تأخير علي؟": [
                "ليش في علي غرامة تأخير؟",
                "ليه حاطين علي غرامة تأخير"
        ],
        "هل يمكنني دفع الرسوم على أقساط؟": [
                "اقدر ادفع الرسوم بالتقسيط؟",
                "ممكن ادفع الرسوم بالتقسيط؟"
        ],
        "طلب تجديد الإقامة معلق منذ ثلاثة أسابيع": [
                "طلب تجديد الاقامة معلق من ٣ اسابيع",
                "طلب تجديد الاقامة صارله ثلاث اسابيع معلق"
        ],
        "كيف يمكنني كفالة والديّ؟": [
                "ابا اكفل الوالد والوالده",
                "كيف اكفل ابوي وامي ؟"
        ],
        "هل الفحص الطبي مطلوب لتجديد الإقامة؟": [
                "هل الفحص الطبي ضروري لتجديد الاقامة؟",
                "هل الفحص الطبي ضروري لجدد اقامتي؟"
        ],
        "تم رفض طلب التأشيرة، ماذا يمكنني أن أفعل؟": [
                "انرفضت الفيزا مالي شو اسوي؟",
                "انرفضت الفيزا حقي شو اسوي؟",
                "الفيزا بتاعتي انرفضت اعمل ايه؟"
        ],
        "هل يمكنني تمديد تأشيرة الزيارة لشهر آخر؟": [
                "اقدر امدد الفيزا لشهر؟",
                "ممكن امدد الفيزا لشهر تاني؟"
        ],
        "لا يمكنني تسجيل الدخول إلى البوابة باستخدام الهوية الرقمية": [
                "مو قادر افتح الهوية الرقمية",
                "الهوية الرقمية مو راضيه تفتح معاي",
                "الهوية الرقمية مش عم تفتح"
        ],
        "التطبيق يتوقف عند رفع مستند": [
                "الابلكيشن مش عم يشتغل لما بدي ارفق الاوراق",
                "مش قادر ارفع المستندات على التطبيق",
                "التطبيق مو شغال لما ابا ارفق المستند",
                "التطبيق يتوقف لما ارفق المستند"
        ],
        "نسيت كلمة المرور ورابط إعادة التعيين لا يصل": [
                "نسيت كلمة السر وماعم يوصلني الرابط لاعمل وحده جديده",
                "نسيت كلمة السر وماوصلني اللنك عشان اغيرها"
        ],
        "الموقع بطيء جداً اليوم": [
                "الويبسايت بطيء اوي",
                "الموقع بطيء كثير"
        ],
        "الاستمارة لا تسمح لي بالتقديم": [
                "الاستمارة ما عم تخليني اقدم",
                "الاستمارة مو شغالة مو قادر اقدم"
        ],
        "الموظفون في مركز الخدمة لم يكونوا متعاونين": [
                "الموظفين بمركز الخدمة ما ساعدوني",
                "الموظفين بالمركز ماقدمولي المساعده المطلوبه",
                "الموظفين فالمركز مساعدونيش"
        ],
        "انتظرت ساعتين ولم يخدمني أحد": [
                "نطرت ساعتين وماحدا ساعدني",
                "استنيت ساعتين ومحدش خدمني",
                "ترييت ساعتين وماشي مساعده",
                "ترييت ساعتين ومافي مساعده"
        ],
        "لا أحد يرد على مركز الاتصال": [
                "محدش بيرد على الكول سنتر",
                "ما حدا عم يرد الى مركز الاتصال",
                "ماشي رد على مكالماتي من مركز الاتصال"
        ],
        "أرغب في تقديم شكوى بشأن طول وقت الانتظار": [
                "بدي قدم شكوى بخصوص وقت الانتظار",
                "ابا اشتكي بخصوص وقت الانتظار"
        ],
        "حصلت على معلومات خاطئة عند الكاونتر": [
                "الموظف عند الكاونتر عطاني معلومات غلط",
                "عطوني معلومات غلط عند الكاونتر"
        ]
    }
    openers = {"en": ["", "Hello, ", "Good morning. ", "Dear team, "], "ar": ["", "مرحباً، ", "هلا، ", "صباح الخير. ", "مساء الخير. ", "السلام عليكم، "]}
    closers = {"en": ["", " Thank you.", " Please help.", " This is urgent."], "ar": ["", " شكراً لكم.", " شكراً.", " أرجو المساعدة.", " ابا مساعدة لو سمحت.", " بدي مساعدة.", " الأمر عاجل.", " الموضوع مستعجل."]}
    units = list(T)
    rows = []
    for i in range(n):
        unit = units[rng.choice(len(units), p=[0.20, 0.20, 0.25, 0.20, 0.15])]
        lang = "ar" if rng.random() < 0.55 else "en"
        core = rng.choice(T[unit][lang])
        if lang == "ar" and rng.random() < 0.6:   # 60% of Arabic requests in dialect, 40% in Modern Standard Arabic
            core = str(rng.choice(AR_DIALECTS[core.rstrip(".")]))
        text = rng.choice(openers[lang]) + core + rng.choice(closers[lang])
        rows.append({"request_id": f"CR-{600000 + i}", "channel": rng.choice(["Email", "Web form", "App", "Call centre notes"]),
                     "language": "Arabic" if lang == "ar" else "English", "text": text, "responsible_unit": unit})
    df = pd.DataFrame(rows)
    # some requests are labelled with the wrong unit, as in real routing logs
    r = np.random.default_rng(seed + 100)
    idx = r.choice(n, int(n * 0.04), replace=False)
    df.loc[idx, "responsible_unit"] = r.choice(units, len(idx))
    return df


if __name__ == "__main__":
    raw, clean = visa_applications()
    raw.to_csv(OUT / "visa_applications_raw.csv", index=False)
    clean.to_csv(OUT / "visa_applications_clean.csv", index=False)
    tax_registrations().to_csv(OUT / "tax_registrations.csv", index=False)
    benefit_claims().to_csv(OUT / "benefit_claims.csv", index=False)
    patients().to_csv(OUT / "patients_service_use.csv", index=False)
    late_applications().to_csv(OUT / "late_applications.csv", index=False)
    citizen_requests().to_csv(OUT / "citizen_requests.csv", index=False, encoding="utf-8")
    for f in sorted(OUT.glob("*.csv")):
        print(f.name, len(pd.read_csv(f)), "rows")
