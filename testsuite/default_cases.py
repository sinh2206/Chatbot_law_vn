from __future__ import annotations

from typing import Any


def _case(
    level: int,
    index: int,
    domain: str,
    question: str,
    expected_answer: str,
    keywords: list[str],
    citations: list[str],
) -> dict[str, Any]:
    return {
        "case_id": f"L{level}_{index:03d}",
        "level": level,
        "domain": domain,
        "question": question,
        "expected_answer": expected_answer,
        "expected_keywords": keywords,
        "expected_citations": citations,
    }


def default_test_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    level_1 = [
        ("HoTich", "Thủ tục đăng ký khai sinh cần giấy tờ gì?", "Nêu hồ sơ cơ bản khi đăng ký khai sinh.", ["khai sinh", "hồ sơ", "đăng ký"], ["60/2014/QH13"]),
        ("HoTich", "Cơ quan nào có thẩm quyền đăng ký kết hôn?", "Trả lời cơ quan đăng ký hộ tịch có thẩm quyền.", ["kết hôn", "thẩm quyền", "ủy ban"], ["123/2015/ND-CP"]),
        ("HoTich", "Đăng ký khai tử nộp ở đâu?", "Nêu nơi tiếp nhận đăng ký khai tử.", ["khai tử", "ủy ban", "cấp xã"], ["60/2014/QH13"]),
        ("HoTich", "Cải chính hộ tịch thực hiện theo quy định nào?", "Chỉ ra văn bản và thủ tục cải chính hộ tịch.", ["cải chính", "hộ tịch", "thủ tục"], ["04/2020/TT-BTP"]),
        ("CCCD", "Khi nào phải đổi thẻ căn cước công dân?", "Nêu các mốc/điều kiện đổi thẻ căn cước.", ["đổi thẻ", "căn cước", "độ tuổi"], ["26/2023/QH15"]),
        ("CCCD", "Làm CCCD lần đầu cần chuẩn bị gì?", "Nêu yêu cầu giấy tờ và thông tin định danh.", ["CCCD", "lần đầu", "thông tin"], ["26/2023/QH15"]),
        ("CCCD", "Thẻ căn cước gắn chip có giá trị thế nào?", "Mô tả giá trị pháp lý của thẻ căn cước.", ["giá trị", "thẻ căn cước", "xác minh"], ["26/2023/QH15"]),
        ("CCCD", "Mất thẻ căn cước thì làm gì?", "Nêu hướng xử lý khi mất thẻ căn cước.", ["mất thẻ", "cấp lại", "căn cước"], ["59/2022/ND-CP"]),
        ("DatDai", "Sổ đỏ là giấy tờ gì?", "Giải thích khái niệm giấy chứng nhận quyền sử dụng đất.", ["sổ đỏ", "giấy chứng nhận", "quyền sử dụng đất"], ["45/2013/QH13"]),
        ("DatDai", "Hộ gia đình có quyền chuyển nhượng đất khi nào?", "Nêu điều kiện cơ bản để chuyển nhượng đất.", ["chuyển nhượng", "đất", "điều kiện"], ["31/2024/QH15"]),
        ("DatDai", "Thu hồi đất do vi phạm được quy định ra sao?", "Tóm tắt nguyên tắc thu hồi đất do vi phạm.", ["thu hồi", "vi phạm", "đất đai"], ["31/2024/QH15"]),
        ("DatDai", "Đăng ký đất đai lần đầu cần lưu ý gì?", "Nêu khâu đăng ký ban đầu và hồ sơ chính.", ["đăng ký", "đất đai", "hồ sơ"], ["43/2014/ND-CP"]),
        ("DoanhNghiep", "Thành lập công ty cần bước nào đầu tiên?", "Nêu bước đăng ký doanh nghiệp cơ bản.", ["thành lập", "đăng ký doanh nghiệp", "hồ sơ"], ["59/2020/QH14"]),
        ("DoanhNghiep", "Tên doanh nghiệp phải đáp ứng điều kiện gì?", "Nêu nguyên tắc đặt tên doanh nghiệp.", ["tên doanh nghiệp", "đặt tên", "không trùng"], ["59/2020/QH14"]),
        ("DoanhNghiep", "Vốn điều lệ có bắt buộc chứng minh không?", "Nêu quy định chung về vốn điều lệ khi đăng ký.", ["vốn điều lệ", "đăng ký", "doanh nghiệp"], ["59/2020/QH14"]),
        ("DoanhNghiep", "Doanh nghiệp tư nhân có tư cách pháp nhân không?", "Trả lời về tư cách pháp nhân doanh nghiệp tư nhân.", ["doanh nghiệp tư nhân", "pháp nhân"], ["59/2020/QH14"]),
        ("Thue", "Kê khai thuế GTGT theo tháng hay quý?", "Nêu nguyên tắc xác định kỳ kê khai thuế GTGT.", ["kê khai", "GTGT", "tháng", "quý"], ["126/2020/ND-CP"]),
        ("Thue", "Doanh nghiệp mới thành lập cần làm thủ tục thuế gì?", "Nêu thủ tục thuế ban đầu cho doanh nghiệp mới.", ["doanh nghiệp mới", "thủ tục thuế", "đăng ký"], ["38/2019/QH14"]),
        ("Thue", "Khi nào phải xuất hóa đơn điện tử?", "Nêu thời điểm lập hóa đơn điện tử.", ["hóa đơn điện tử", "lập hóa đơn", "thời điểm"], ["80/2021/TT-BTC"]),
        ("Thue", "Khấu trừ thuế đầu vào cần điều kiện gì?", "Nêu điều kiện cơ bản để được khấu trừ thuế.", ["khấu trừ", "thuế đầu vào", "điều kiện"], ["80/2021/TT-BTC"]),
    ]

    level_2 = [
        ("HoTich", "Trình tự đăng ký lại khai sinh gồm những bước nào?", "Tổng hợp trình tự và thành phần hồ sơ đăng ký lại khai sinh.", ["đăng ký lại", "khai sinh", "trình tự", "hồ sơ"], ["123/2015/ND-CP", "04/2020/TT-BTP"]),
        ("HoTich", "Thay đổi cải chính hộ tịch cho người chưa thành niên cần điều kiện gì?", "Nêu điều kiện và chủ thể có quyền yêu cầu.", ["thay đổi", "cải chính", "chưa thành niên"], ["60/2014/QH13", "04/2020/TT-BTP"]),
        ("HoTich", "Đăng ký nhận cha mẹ con cần chứng cứ nào?", "Nêu yêu cầu chứng cứ chứng minh quan hệ huyết thống.", ["nhận cha mẹ con", "chứng cứ", "hộ tịch"], ["123/2015/ND-CP"]),
        ("HoTich", "Đăng ký hộ tịch trực tuyến cần lưu ý gì về đối chiếu giấy tờ?", "Nêu yêu cầu kiểm tra hồ sơ bản giấy/bản điện tử.", ["trực tuyến", "đối chiếu", "giấy tờ"], ["87/2020/ND-CP"]),
        ("CCCD", "Cấp lại CCCD do sai thông tin gồm các bước nào?", "Nêu quy trình xử lý cấp lại do sai thông tin.", ["cấp lại", "sai thông tin", "CCCD"], ["59/2022/ND-CP", "26/2023/QH15"]),
        ("CCCD", "Tích hợp thông tin vào thẻ căn cước thực hiện thế nào?", "Tóm tắt nguyên tắc tích hợp thông tin định danh.", ["tích hợp", "thông tin", "thẻ căn cước"], ["26/2023/QH15"]),
        ("CCCD", "So sánh quy định cũ CMND và căn cước mới về giá trị sử dụng", "Nêu điểm khác biệt chính giữa giấy tờ định danh cũ/mới.", ["CMND", "căn cước", "giá trị sử dụng"], ["59/2014/QH13", "26/2023/QH15"]),
        ("CCCD", "Thủ tục cấp thẻ cho người dưới 14 tuổi có điểm gì đặc thù?", "Nêu yêu cầu riêng cho nhóm chưa đủ tuổi.", ["dưới 14", "thủ tục", "căn cước"], ["26/2023/QH15"]),
        ("DatDai", "Điều kiện tách thửa theo khung pháp luật đất đai mới là gì?", "Tổng hợp điều kiện tách thửa theo luật và nghị định hướng dẫn.", ["tách thửa", "điều kiện", "đất đai"], ["31/2024/QH15", "43/2014/ND-CP"]),
        ("DatDai", "Trình tự chuyển mục đích sử dụng đất gồm các bước nào?", "Nêu hồ sơ, thẩm quyền và nghĩa vụ tài chính.", ["chuyển mục đích", "sử dụng đất", "nghĩa vụ tài chính"], ["31/2024/QH15", "148/2020/ND-CP"]),
        ("DatDai", "So sánh cấp sổ lần đầu và đăng ký biến động đất đai", "Phân biệt hồ sơ và trình tự của hai thủ tục.", ["cấp sổ", "đăng ký biến động", "hồ sơ"], ["43/2014/ND-CP"]),
        ("DatDai", "Khi tranh chấp đất, hồ sơ hòa giải ở cấp xã cần gì?", "Nêu thành phần hồ sơ và đầu mối tiếp nhận.", ["tranh chấp", "hòa giải", "cấp xã"], ["45/2013/QH13", "31/2024/QH15"]),
        ("DoanhNghiep", "Hồ sơ đăng ký công ty TNHH một thành viên gồm gì?", "Nêu thành phần hồ sơ và thông tin bắt buộc.", ["TNHH", "hồ sơ đăng ký", "thành viên"], ["59/2020/QH14"]),
        ("DoanhNghiep", "Quy trình thay đổi người đại diện pháp luật ra sao?", "Nêu nghĩa vụ đăng ký thay đổi nội dung doanh nghiệp.", ["người đại diện", "thay đổi", "đăng ký"], ["59/2020/QH14", "168/2025/ND-CP"]),
        ("DoanhNghiep", "Giải thể doanh nghiệp cần thanh toán nghĩa vụ theo thứ tự nào?", "Nêu thứ tự xử lý nghĩa vụ khi giải thể.", ["giải thể", "thanh toán", "nghĩa vụ"], ["59/2020/QH14"]),
        ("DoanhNghiep", "Điều kiện tạm ngừng kinh doanh và thông báo thế nào?", "Nêu thời hạn và cách thức thông báo tạm ngừng.", ["tạm ngừng", "thông báo", "kinh doanh"], ["59/2020/QH14"]),
        ("Thue", "Điều kiện khai thuế theo quý đối với doanh nghiệp nhỏ là gì?", "Nêu tiêu chí và căn cứ xác định khai quý.", ["khai thuế", "theo quý", "doanh nghiệp nhỏ"], ["126/2020/ND-CP"]),
        ("Thue", "Trình tự điều chỉnh hóa đơn sai sót theo thông tư hiện hành", "Nêu cách xử lý hóa đơn sai và lập chứng từ điều chỉnh.", ["hóa đơn sai sót", "điều chỉnh", "thông tư"], ["80/2021/TT-BTC", "86/2024/TT-BTC"]),
        ("Thue", "Thủ tục hoàn thuế GTGT cơ bản gồm các bước nào?", "Nêu hồ sơ hoàn thuế và quy trình xử lý.", ["hoàn thuế", "GTGT", "hồ sơ"], ["38/2019/QH14", "80/2021/TT-BTC"]),
        ("Thue", "Nghĩa vụ lưu trữ chứng từ kế toán phục vụ kê khai thuế", "Nêu yêu cầu lưu trữ hóa đơn/chứng từ.", ["lưu trữ", "chứng từ", "kê khai thuế"], ["38/2019/QH14"]),
    ]

    level_3 = [
        ("DoanhNghiep,Thue", "Sau khi thành lập công ty, cần làm gì để đăng ký thuế ban đầu?", "Kết hợp thủ tục đăng ký doanh nghiệp và nghĩa vụ thuế ban đầu.", ["thành lập", "đăng ký thuế", "mã số thuế"], ["59/2020/QH14", "126/2020/ND-CP"]),
        ("DoanhNghiep,Thue", "Giải thể doanh nghiệp thì nghĩa vụ thuế cần xử lý ra sao?", "Nêu đồng thời thủ tục giải thể và quyết toán/đóng mã số thuế.", ["giải thể", "quyết toán", "thuế"], ["59/2020/QH14", "38/2019/QH14"]),
        ("DoanhNghiep,DatDai", "Doanh nghiệp thuê đất để mở nhà xưởng cần thủ tục nào?", "Kết hợp thủ tục doanh nghiệp và quyền sử dụng đất.", ["thuê đất", "nhà xưởng", "thủ tục"], ["59/2020/QH14", "31/2024/QH15"]),
        ("DatDai,Thue", "Chuyển nhượng đất phải kê khai nghĩa vụ tài chính gì?", "Nêu nghĩa vụ tài chính thuế/phi phí khi chuyển nhượng đất.", ["chuyển nhượng đất", "nghĩa vụ tài chính", "kê khai"], ["31/2024/QH15", "38/2019/QH14"]),
        ("CCCD,HoTich", "Đăng ký khai sinh xong thì cập nhật định danh cá nhân thế nào?", "Nêu liên hệ giữa hộ tịch và dữ liệu căn cước.", ["khai sinh", "định danh cá nhân", "cập nhật"], ["60/2014/QH13", "26/2023/QH15"]),
        ("CCCD,HoTich", "Khi thay đổi thông tin hộ tịch, có phải đổi CCCD không?", "Nêu trường hợp cần cập nhật thông tin trên thẻ căn cước.", ["thay đổi hộ tịch", "đổi CCCD", "cập nhật"], ["04/2020/TT-BTP", "26/2023/QH15"]),
        ("DoanhNghiep,CCCD", "Mở doanh nghiệp tư nhân có cần người đại diện dùng CCCD còn hạn không?", "Nêu yêu cầu về giấy tờ định danh khi đăng ký doanh nghiệp.", ["doanh nghiệp tư nhân", "CCCD", "đăng ký"], ["59/2020/QH14", "26/2023/QH15"]),
        ("DoanhNghiep,HoTich", "Thay đổi họ tên cá nhân sáng lập ảnh hưởng hồ sơ doanh nghiệp thế nào?", "Nêu việc cập nhật thông tin chủ thể sau thay đổi hộ tịch.", ["thay đổi họ tên", "hồ sơ doanh nghiệp", "cập nhật"], ["60/2014/QH13", "59/2020/QH14"]),
        ("DatDai,DoanhNghiep", "Doanh nghiệp nhận chuyển nhượng đất cần các bước pháp lý nào?", "Kết hợp điều kiện nhận chuyển nhượng và đăng ký biến động.", ["nhận chuyển nhượng", "đăng ký biến động", "doanh nghiệp"], ["31/2024/QH15", "43/2014/ND-CP"]),
        ("DatDai,Thue", "Khi chuyển mục đích đất sang thương mại có phát sinh thuế gì?", "Nêu nghĩa vụ tài chính kèm chuyển mục đích sử dụng đất.", ["chuyển mục đích", "đất thương mại", "thuế"], ["31/2024/QH15", "38/2019/QH14"]),
        ("Thue,DoanhNghiep", "Công ty tạm ngừng kinh doanh thì có phải nộp tờ khai thuế không?", "Nêu nghĩa vụ khai thuế khi tạm ngừng.", ["tạm ngừng", "tờ khai thuế", "công ty"], ["59/2020/QH14", "126/2020/ND-CP"]),
        ("HoTich,CCCD", "Đăng ký khai tử có ảnh hưởng thế nào đến dữ liệu căn cước?", "Nêu nguyên tắc đồng bộ dữ liệu hộ tịch - căn cước.", ["khai tử", "dữ liệu căn cước", "đồng bộ"], ["60/2014/QH13", "26/2023/QH15"]),
        ("DoanhNghiep,Thue", "Chuyển đổi loại hình doanh nghiệp có cần điều chỉnh đăng ký thuế không?", "Nêu mối liên hệ hồ sơ thay đổi doanh nghiệp và thuế.", ["chuyển đổi loại hình", "điều chỉnh thuế", "đăng ký"], ["59/2020/QH14", "126/2020/ND-CP"]),
        ("DatDai,DoanhNghiep", "Góp vốn bằng quyền sử dụng đất vào công ty cần thủ tục gì?", "Nêu điều kiện góp vốn bằng đất và đăng ký biến động.", ["góp vốn", "quyền sử dụng đất", "công ty"], ["31/2024/QH15", "59/2020/QH14"]),
        ("HoTich,DoanhNghiep", "Người đại diện đổi giấy tờ hộ tịch thì doanh nghiệp cần cập nhật gì?", "Nêu nghĩa vụ cập nhật thông tin đăng ký doanh nghiệp.", ["người đại diện", "đổi giấy tờ", "cập nhật"], ["04/2020/TT-BTP", "59/2020/QH14"]),
        ("CCCD,Thue", "Cá nhân kinh doanh dùng CCCD gắn chip để khai thuế điện tử thế nào?", "Nêu điều kiện xác thực định danh và khai thuế.", ["cá nhân kinh doanh", "CCCD gắn chip", "khai thuế điện tử"], ["26/2023/QH15", "38/2019/QH14"]),
        ("DatDai,HoTich", "Thay đổi tình trạng hôn nhân có ảnh hưởng thủ tục sang tên sổ đỏ không?", "Nêu khía cạnh giấy tờ nhân thân trong thủ tục đất đai.", ["tình trạng hôn nhân", "sang tên sổ đỏ", "giấy tờ"], ["60/2014/QH13", "31/2024/QH15"]),
        ("DoanhNghiep,Thue", "Chấm dứt hoạt động chi nhánh thì thủ tục doanh nghiệp và thuế thế nào?", "Nêu quy trình kết hợp chấm dứt chi nhánh và nghĩa vụ thuế.", ["chi nhánh", "chấm dứt hoạt động", "thuế"], ["59/2020/QH14", "126/2020/ND-CP"]),
        ("Thue,DatDai", "Thuế liên quan hợp đồng thuê đất trả tiền hàng năm gồm gì?", "Nêu sắc thuế và nghĩa vụ tài chính thường gặp.", ["thuê đất", "trả tiền hàng năm", "nghĩa vụ thuế"], ["31/2024/QH15", "38/2019/QH14"]),
        ("CCCD,DoanhNghiep", "Hộ kinh doanh lên doanh nghiệp có cần cập nhật định danh gì?", "Nêu yêu cầu định danh chủ thể khi chuyển đổi.", ["hộ kinh doanh", "lên doanh nghiệp", "định danh"], ["26/2023/QH15", "59/2020/QH14"]),
    ]

    level_4 = [
        ("DatDai", "Tôi muốn làm sổ đỏ thì cần những gì?", "Yêu cầu chatbot hỏi rõ loại đất, nguồn gốc đất và địa phương trước khi kết luận.", ["làm sổ đỏ", "loại đất", "hồ sơ"], ["31/2024/QH15"]),
        ("Thue", "Thông tư cũ về hóa đơn còn áp dụng không?", "Yêu cầu xác định hiệu lực văn bản và nêu văn bản thay thế.", ["hiệu lực", "hóa đơn", "văn bản thay thế"], ["80/2021/TT-BTC", "86/2024/TT-BTC"]),
        ("DoanhNghiep,Thue", "Doanh nghiệp tôi mới giải thể xong có phải nộp gì nữa không?", "Trả lời theo hướng cần thông tin thời điểm và trạng thái quyết toán.", ["giải thể", "quyết toán", "nghĩa vụ còn lại"], ["59/2020/QH14", "38/2019/QH14"]),
        ("DatDai", "Luật đất đai 2013 và 2024 khác nhau điểm nào khi cấp sổ?", "So sánh văn bản cũ/mới và cảnh báo áp dụng theo thời điểm hiệu lực.", ["2013", "2024", "cấp sổ"], ["45/2013/QH13", "31/2024/QH15"]),
        ("CCCD", "CMND 9 số của tôi còn dùng được mãi không?", "Nêu yêu cầu chuyển đổi giấy tờ định danh theo quy định mới.", ["CMND 9 số", "chuyển đổi", "căn cước"], ["26/2023/QH15", "59/2014/QH13"]),
        ("HoTich", "Muốn sửa năm sinh trên giấy khai sinh thì làm sao?", "Yêu cầu làm rõ căn cứ chứng minh và cơ quan giải quyết.", ["sửa năm sinh", "cải chính", "chứng minh"], ["60/2014/QH13", "04/2020/TT-BTP"]),
        ("Thue", "Nghị định thuế cũ hết hiệu lực thì tra ở đâu?", "Kích hoạt fallback và trả nguồn chính thống.", ["hết hiệu lực", "tra cứu", "nguồn chính thống"], ["126/2020/ND-CP"]),
        ("DoanhNghiep", "Tên công ty giống thương hiệu nổi tiếng có được không?", "Trả lời thận trọng, phân biệt pháp luật doanh nghiệp và sở hữu trí tuệ.", ["tên công ty", "không trùng", "nhầm lẫn"], ["59/2020/QH14"]),
        ("DatDai,Thue", "Tôi bán đất nhưng chưa sang tên thì đóng thuế ra sao?", "Nêu cần làm rõ tình trạng hợp đồng/công chứng/đăng ký.", ["bán đất", "chưa sang tên", "thuế"], ["31/2024/QH15", "38/2019/QH14"]),
        ("HoTich,CCCD", "Sai tên trên CCCD khác với khai sinh thì sửa bên nào trước?", "Nêu quy trình ưu tiên chỉnh hộ tịch gốc rồi cập nhật căn cước.", ["sai tên", "CCCD", "khai sinh"], ["04/2020/TT-BTP", "26/2023/QH15"]),
        ("DoanhNghiep,DatDai", "Công ty thuê đất nông nghiệp để làm nhà xưởng được không?", "Yêu cầu phân tích mục đích sử dụng đất và điều kiện chuyển mục đích.", ["thuê đất nông nghiệp", "nhà xưởng", "chuyển mục đích"], ["31/2024/QH15", "59/2020/QH14"]),
        ("Thue", "Hóa đơn tháng trước xuất sai mà nay mới phát hiện xử lý thế nào?", "Yêu cầu nêu quy trình điều chỉnh/thay thế theo văn bản hiện hành.", ["hóa đơn sai", "điều chỉnh", "thay thế"], ["80/2021/TT-BTC", "86/2024/TT-BTC"]),
        ("DatDai", "Tôi có đất không giấy tờ từ lâu thì có làm sổ được không?", "Nêu cần xác minh nguồn gốc, thời điểm sử dụng, xác nhận địa phương.", ["không giấy tờ", "nguồn gốc", "xác nhận"], ["31/2024/QH15", "43/2014/ND-CP"]),
        ("DoanhNghiep", "Công ty ngừng hoạt động lâu chưa báo, giờ xử lý sao?", "Nêu rủi ro pháp lý và nghĩa vụ khắc phục hồ sơ đăng ký.", ["ngừng hoạt động", "không báo", "khắc phục"], ["59/2020/QH14"]),
        ("CCCD", "Tôi đổi nơi thường trú thì có bắt buộc đổi CCCD không?", "Nêu khi nào cần cập nhật thông tin cư trú trên thẻ.", ["đổi nơi thường trú", "cập nhật", "CCCD"], ["26/2023/QH15"]),
        ("HoTich", "Con sinh ở nước ngoài về Việt Nam đăng ký lại thế nào?", "Nêu thủ tục có yếu tố nước ngoài và hồ sơ chứng minh.", ["nước ngoài", "đăng ký lại", "hộ tịch"], ["123/2015/ND-CP", "04/2020/TT-BTP"]),
        ("DoanhNghiep,Thue", "Ngừng kinh doanh tạm thời có cần nộp lệ phí môn bài không?", "Nêu điều kiện miễn/không miễn theo thời gian ngừng.", ["tạm ngừng", "lệ phí môn bài", "điều kiện"], ["59/2020/QH14", "126/2020/ND-CP"]),
        ("DatDai", "Sổ đỏ cũ và sổ hồng khác gì khi giao dịch?", "Giải thích giá trị pháp lý và tránh nhầm lẫn thuật ngữ.", ["sổ đỏ", "sổ hồng", "giá trị pháp lý"], ["45/2013/QH13", "31/2024/QH15"]),
        ("CCCD,Thue", "Dùng VNeID để làm thủ tục thuế có đủ chưa?", "Nêu yêu cầu định danh điện tử và hệ thống thuế hỗ trợ.", ["VNeID", "thủ tục thuế", "định danh điện tử"], ["26/2023/QH15", "38/2019/QH14"]),
        ("DoanhNghiep,HoTich", "Chủ doanh nghiệp qua đời thì công ty xử lý pháp lý ra sao?", "Nêu trường hợp thừa kế/quản trị và cập nhật đăng ký.", ["chủ doanh nghiệp qua đời", "thừa kế", "đăng ký"], ["59/2020/QH14", "60/2014/QH13"]),
    ]

    for index, item in enumerate(level_1, start=1):
        cases.append(_case(1, index, *item))

    for index, item in enumerate(level_2, start=1):
        cases.append(_case(2, index, *item))

    for index, item in enumerate(level_3, start=1):
        cases.append(_case(3, index, *item))

    for index, item in enumerate(level_4, start=1):
        cases.append(_case(4, index, *item))

    return cases
