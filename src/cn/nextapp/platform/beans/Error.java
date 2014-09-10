package cn.nextapp.platform.beans;

/**
 * 错误码表
 * @author Winter Lau
 * @date 2012-1-14 下午12:27:30
 */
public class Error extends Pojo {

	private final static Error INSTANCE = new Error();
	
	public final static byte TYPE_AUDIT = 0x01; //审批错误
	public final static byte TYPE_BUILD = 0x02;	//构建错误
	
	public static String getErrorMsg(int error_code) {
		Error err = INSTANCE.Get(error_code);
		return (err!=null)?err.getDetail():null;
	}
	
	public static Error auditError(String msg) {
		return _Error(TYPE_AUDIT, msg);
	}

	public static Error buildError(String msg) {
		return _Error(TYPE_BUILD, msg);
	}

	private static Error _Error(byte type, String msg) {
		Error err = new Error();
		err.setDetail(msg);
		err.setLevel((byte)1);
		err.setType(type);
		err.Save();
		return err;
	}
	
	private byte type;
	private byte level;
	private String detail;
	
	public byte getType() {
		return type;
	}
	public void setType(byte type) {
		this.type = type;
	}
	public byte getLevel() {
		return level;
	}
	public void setLevel(byte level) {
		this.level = level;
	}
	public String getDetail() {
		return detail;
	}
	public void setDetail(String detail) {
		this.detail = detail;
	}
	
}
