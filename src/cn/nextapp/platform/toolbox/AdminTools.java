package cn.nextapp.platform.toolbox;

import java.util.List;

import javax.servlet.http.HttpServletRequest;

import cn.nextapp.platform.SmtpHelper;
import cn.nextapp.platform.beans.App;

/**
 * 后台管理工具包
 * @author Winter Lau
 * @date 2011-12-30 下午5:47:01
 */
public class AdminTools {

	public static int app_count_by_status(int status) {
		return App.CountByStatus(status);
	}
	
	public static List<App> list_app_by_status(int status, int page, int count) {
		return App.ListByStatus(status, page, count);
	}
	
	/**
	 * 报告错误信息
	 * @param t
	 */
	public static void report_error(HttpServletRequest req){
		SmtpHelper.reportError(req, null);
	}
	
	public static Throwable get_exception(HttpServletRequest req) {
		return SmtpHelper.GetException(req);
	}

}
